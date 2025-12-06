# file: app.py
"""
Simple incremental learning service (Flask).
Run: pip install flask apscheduler pandas
Start: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
import os, csv, json, datetime, threading
from collections import Counter, defaultdict
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'draws.csv')
LOG_PATH = os.path.join(DATA_DIR, 'learning_logs.jsonl')
CACHE_PATH = os.path.join(DATA_DIR, 'cache.json')

# model hyperparams
EMA_ALPHA = 0.3           # how fast EMA adapts for probabilities
MOMENTUM_WINDOW_DAYS = 30
TOP_N = 6

# Ensure data dir + files
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('date,region,number\n')
if not os.path.exists(LOG_PATH):
    open(LOG_PATH, 'a').close()
if not os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'w') as f:
        json.dump({'regions': {}, 'last_updated': None}, f)

app = Flask(__name__)

# --- Utilities ---
def load_csv():
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    # normalize number formatting (zero-pad)
    df['number'] = df['number'].astype(str).str.zfill(2)
    df['region'] = df['region'].astype(str).str.upper()
    return df

def append_draw(date_str, region, number):
    """Append a new draw to CSV and perform incremental update synchronously."""
    region = str(region).upper()
    number = str(number).zfill(2)
    date_iso = date_str
    # append row
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date_iso, region, number])
    # update model for this month and region + ALL
    dt = datetime.datetime.fromisoformat(date_iso)
    month = dt.month
    # perform incremental update for target month & regions
    for r in (region, 'ALL'):
        incremental_update(month, r, note='append_draw')
    return True

def tail_logs(n=20):
    with open(LOG_PATH, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines[-n:]

def save_cache_for(region, month, entry):
    with threading.Lock():
        with open(CACHE_PATH, 'r', encoding='utf8') as f:
            cache = json.load(f)
        cache.setdefault('regions', {}).setdefault(region, {})[str(month)] = entry
        cache['last_updated'] = datetime.datetime.utcnow().isoformat()
        with open(CACHE_PATH, 'w', encoding='utf8') as f:
            json.dump(cache, f, indent=2)

def read_cache():
    with open(CACHE_PATH, 'r', encoding='utf8') as f:
        return json.load(f)

def write_log(entry):
    with open(LOG_PATH, 'a', encoding='utf8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# --- Core modeling functions ---
def compute_month_freq(df, month, region):
    mdf = df[df['date'].dt.month == int(month)]
    if region != 'ALL':
        mdf = mdf[mdf['region'] == region]
    counts = Counter(mdf['number'].tolist())
    return counts, len(mdf)

def compute_momentum(df, region, days=MOMENTUM_WINDOW_DAYS):
    cutoff = pd.Timestamp('now') - pd.Timedelta(days=days)
    mdf = df[df['date'] >= cutoff]
    if region != 'ALL':
        mdf = mdf[mdf['region'] == region]
    counts = Counter(mdf['number'].tolist())
    return counts

def merge_with_ema(prev_probs, freq_counts, total, ema_alpha=EMA_ALPHA):
    # prev_probs: dict number->prob, freq_counts: Counter number->count, total: observed total for freq_counts
    # produce new probs as EMA between prev and current freq-based probs
    curr_probs = {}
    if total <= 0:
        # no new data; keep prev
        return prev_probs.copy() if prev_probs else {}
    for num, cnt in freq_counts.items():
        curr_probs[num] = cnt / total
    # include prev keys too
    all_nums = set(prev_probs.keys()) | set(curr_probs.keys())
    new_probs = {}
    for num in all_nums:
        p_prev = prev_probs.get(num, 0.0)
        p_curr = curr_probs.get(num, 0.0)
        new_probs[num] = round((1-ema_alpha)*p_prev + ema_alpha*p_curr, 4)
    # normalize
    s = sum(new_probs.values()) or 1.0
    for k in list(new_probs.keys()):
        new_probs[k] = round(new_probs[k]/s, 4)
    return new_probs

def build_rationale(month, region, sample_size, momentum_counts):
    reasons = [
        f"Seasonal frequency for month {month} in region {region}",
        f"Sample size {sample_size}",
    ]
    if sum(momentum_counts.values())>0:
        reasons.append(f"Momentum from last {MOMENTUM_WINDOW_DAYS} days")
    else:
        reasons.append("No strong recent momentum")
    # detect simple digit bias (ending digit)
    endings = Counter([n[-1] for n in momentum_counts.keys()]) if momentum_counts else Counter()
    if endings:
        most_common_end, c = endings.most_common(1)[0]
        if c >= 3:
            reasons.append(f"Digit bias favored ending with {most_common_end}")
    return reasons

def incremental_update(month, region, note='mini'):
    """Compute/update cached probabilities for month+region using EMA + frequency + momentum.
       Also append log entry to learning_logs.jsonl."""
    df = load_csv()
    freq_counts, sample_size = compute_month_freq(df, month, region)
    momentum_counts = compute_momentum(df, region)
    # read previous probs from cache if exist
    cache = read_cache()
    prev = cache.get('regions', {}).get(region, {}).get(str(month), {}).get('top_probabilities', [])
    prev_probs = {p['number']: p['prob'] for p in prev} if prev else {}
    # merge using EMA
    # if no frequency data for this month, rely on prev + momentum fallback
    if sample_size > 0:
        merged = merge_with_ema(prev_probs, freq_counts, sum(freq_counts.values()))
    else:
        # use momentum as pseudo-sample
        merged = merge_with_ema(prev_probs, momentum_counts, sum(momentum_counts.values()) or 1)
    # produce sorted top list
    sorted_probs = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[:TOP_N]
    top_probabilities = [{'number': num, 'prob': round(float(prob), 4)} for num, prob in sorted_probs]
    rationale = build_rationale(month, region, sample_size, momentum_counts)
    confidence = top_probabilities[0]['prob'] if top_probabilities else 0.0
    entry = {
        'month': int(month),
        'region': region,
        'top_probabilities': top_probabilities,
        'rationale': rationale,
        'alpha': [p['number'] for p in top_probabilities[:2]],
        'last_updated': datetime.datetime.utcnow().isoformat(),
        'confidence': round(float(confidence), 4),
        'note': note
    }
    save_cache_for(region, month, entry)
    log_entry = {'ts': datetime.datetime.utcnow().isoformat(), 'month': month, 'region': region,
                 'top': top_probabilities, 'rationale': rationale, 'confidence': confidence, 'note': note}
    write_log(log_entry)
    return entry

def build_ensemble(month, region):
    month = int(month)
    region = str(region).upper()
    cache = read_cache()
    entry = cache.get('regions', {}).get(region, {}).get(str(month))
    if entry:
        # attach recent results summary
        df = load_csv()
        today = pd.Timestamp('today').normalize()
        yesterday = today - pd.Timedelta(days=1)
        recent = {
            'today': df[df['date'] >= today]['number'].unique().tolist(),
            'yesterday': df[(df['date'] >= yesterday) & (df['date'] < today)]['number'].unique().tolist(),
            'month': defaultdict(list)
        }
        # monthwise group for current month
        mdf = df[df['date'].dt.month == month]
        for _, r in mdf.groupby(mdf['date'].dt.date):
            d = str(r['date'].iloc[0].date())
            recent['month'][d] = r['number'].tolist()
        entry_out = {
            'month': month,
            'region': region,
            'alpha': entry.get('alpha', []),
            'angel': [p['number'] for p in entry.get('top_probabilities', [])[2:4]],
            'dominated': [p['number'] for p in entry.get('top_probabilities', [])[4:6]],
            'top_probabilities': entry.get('top_probabilities', []),
            'rationale': entry.get('rationale', []),
            'confidence': entry.get('confidence', 0.0),
            'recent_results': recent,
            'last_updated': entry.get('last_updated')
        }
        return entry_out
    else:
        # if cache miss, compute on the fly and store
        return incremental_update(month, region, note='on-demand')

# --- Flask endpoints ---
@app.route('/insight', methods=['GET'])
def insight():
    month = int(request.args.get('month') or (datetime.datetime.utcnow().month))
    region = (request.args.get('region') or 'ALL').upper()
    e = build_ensemble(month, region)
    return jsonify(e)

@app.route('/draw', methods=['POST'])
def add_draw():
    payload = request.get_json(force=True)
    # expect { "date": "YYYY-MM-DDTHH:MM:SS", "region":"G", "number":"36" }
    date = payload.get('date') or datetime.datetime.utcnow().isoformat()
    region = payload.get('region') or 'ALL'
    number = payload.get('number')
    if not number:
        return jsonify({'error': 'number required'}), 400
    try:
        append_draw(date, region, number)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    month = datetime.datetime.fromisoformat(date).month
    return jsonify(build_ensemble(month, region))

@app.route('/admin/latest-log', methods=['GET'])
def latest_log():
    lines = tail_logs(50)
    return app.response_class('\n'.join(lines), mimetype='text/plain')

# serve static if needed
@app.route('/')
def root():
    return jsonify({"msg": "Python incremental learner running. Use /insight and /draw endpoints."})

# --- Scheduler (optional background retrain) ---
def mini_retrain_all():
    now_iso = datetime.datetime.utcnow().isoformat()
    month = datetime.datetime.utcnow().month
    for region in ['G','F','D','ALL']:
        incremental_update(month, region, note='mini-scheduled')

if __name__ == '__main__':
    # run a first-time full pass for current month to populate cache (mirrors the Node behaviour). 
    try:
        mini_retrain_all()
    except Exception:
        pass
    # schedule periodic mini retrain every 20 minutes (configurable)
    scheduler = BackgroundScheduler()
    scheduler.add_job(mini_retrain_all, 'interval', minutes=20, id='mini_retrain')
    scheduler.start()
    app.run(host='0.0.0.0', port=3000, debug=False)