import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


prediction_requests_total = Counter(
    "prediction_requests_total",
    "Nombre total de requêtes de prédiction"
)


prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Temps de traitement d'une prédiction en secondes"
)


prediction_failures_total = Counter(
    "prediction_failures_total",
    "Nombre de requêtes de prédiction échouées"
)

# 4. Santé du backend (1 = up)
backend_up = Gauge(
    "backend_up",
    "Statut du backend (1 = opérationnel)"
)
backend_up.set(1)

_start_time = time.time()
backend_uptime_seconds = Gauge(
    "backend_uptime_seconds",
    "Uptime du backend en secondes"
)

def get_metrics():
    backend_uptime_seconds.set(time.time() - _start_time)
    return generate_latest(), CONTENT_TYPE_LATEST