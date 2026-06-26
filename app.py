"""
PhiURL - Phishing URL Detection API
Flask backend — production-ready with full CORS + proper error handling
"""
import os
import re
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# ── CORS: allow ANY origin so the HTML file works when opened directly ──────────
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Load ML models ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model      = joblib.load(os.path.join(BASE_DIR, "logistic_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    print("✓ Models loaded successfully")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found: {e}.\n"
        f"Place logistic_model.pkl and tfidf_vectorizer.pkl in: {BASE_DIR}"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────────
SUSPICIOUS_TLDS   = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.ru', '.cn', '.top', '.click', '.loan', '.work'}
SUSPICIOUS_WORDS  = ['login', 'verify', 'secure', 'update', 'confirm', 'account', 'banking', 'paypal', 'password', 'credential']
IP_PATTERN        = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

def extract_domain_features(url: str) -> dict:
    """Extract heuristic domain features for richer frontend display."""
    try:
        parsed   = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed.hostname or ''
        path     = parsed.path or ''

        tld = '.' + hostname.split('.')[-1] if '.' in hostname else ''
        suspicious_tld   = tld.lower() in SUSPICIOUS_TLDS
        suspicious_words = any(w in url.lower() for w in SUSPICIOUS_WORDS)
        is_ip_url        = bool(IP_PATTERN.match(url))
        has_subdomain    = hostname.count('.') >= 2
        long_url         = len(url) > 75
        has_https        = url.startswith('https://')
        dash_in_domain   = '-' in hostname.split('.')[0] if hostname else False
        double_slash     = '//' in path
        at_symbol        = '@' in url

        risk_score = sum([
            suspicious_tld * 25,
            suspicious_words * 20,
            is_ip_url * 30,
            long_url * 10,
            dash_in_domain * 10,
            double_slash * 15,
            at_symbol * 20,
            not has_https * 5,
        ])

        return {
            "hostname":        hostname,
            "tld":             tld,
            "has_https":       has_https,
            "suspicious_tld":  suspicious_tld,
            "is_ip_based":     is_ip_url,
            "has_subdomain":   has_subdomain,
            "long_url":        long_url,
            "dash_in_domain":  dash_in_domain,
            "heuristic_risk":  min(risk_score, 100),
        }
    except Exception:
        return {}


# ── Routes ───────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "PhiURL API", "models_loaded": True})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Main prediction endpoint — returns prediction + confidence + beliefs + features."""
    if request.method == "OPTIONS":          # preflight
        return jsonify({}), 200

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        url = data.get("url", "")
        if not url or not isinstance(url, str) or not url.strip():
            return jsonify({"error": "Missing or empty 'url' field"}), 400

        url = url.strip()

        # ── ML prediction ──────────────────────────────────────────────────
        X      = vectorizer.transform([url])
        pred   = model.predict(X)[0]
        proba  = model.predict_proba(X)[0]

        # proba[0] = P(Legitimate),  proba[1] = P(Phishing)
        p_legitimate = round(float(proba[0]) * 100, 2)
        p_phishing   = round(float(proba[1]) * 100, 2)
        confidence   = round(float(max(proba)) * 100, 2)

        # ── Belief scores (LBP-inspired influence decomposition) ───────────
        weights       = model.coef_[0]
        feature_vals  = X.toarray()[0]
        influence     = weights * feature_vals

        belief_scores = {
            "suspicious_tokens": float(np.sum(influence[influence > 0])),
            "benign_structure":  float(abs(np.sum(influence[influence < 0]))),
        }
        total = belief_scores["suspicious_tokens"] + belief_scores["benign_structure"] + 1e-6
        beliefs = {
            "Suspicious Pattern Belief": round((belief_scores["suspicious_tokens"] / total) * 100, 2),
            "Benign Structure Belief":   round((belief_scores["benign_structure"]  / total) * 100, 2),
        }

        # ── Domain heuristics ──────────────────────────────────────────────
        features = extract_domain_features(url)

        return jsonify({
            "prediction":    "Phishing" if pred == 1 else "Legitimate",
            "confidence":    confidence,
            "p_legitimate":  p_legitimate,   # always P(Legitimate) — no guessing on frontend
            "p_phishing":    p_phishing,     # always P(Phishing)
            "beliefs":       beliefs,
            "features":      features,
            "url_scanned":   url,
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


if __name__ == "__main__":
    print("\n🛡  PhiURL API starting...")
    print(f"   Models dir : {BASE_DIR}")
    print(f"   Running on : http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
