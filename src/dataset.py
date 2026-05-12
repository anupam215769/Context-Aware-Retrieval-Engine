"""
dataset.py
----------
Built-in corpus of 10 technical paragraphs covering distributed systems topics.
Each document has: id, title, category, and content fields.

These paragraphs are carefully written so that semantic search can distinguish
between related but distinct topics (load balancing vs caching, auth vs monitoring, etc.)
"""

TECHNICAL_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Load Balancing & Auto-Scaling for Peak Traffic",
        "category": "infrastructure",
        "content": (
            "The system employs horizontal auto-scaling combined with a round-robin load balancer "
            "to handle peak traffic surges. During high-demand periods, the orchestrator — built on "
            "Kubernetes HPA (Horizontal Pod Autoscaler) — monitors CPU utilization and request-per-second "
            "metrics via Prometheus. When average CPU exceeds 70% for two consecutive minutes, new compute "
            "nodes are provisioned from a pre-warmed instance pool within 30 seconds. The load balancer "
            "distributes incoming requests across all healthy nodes using weighted round-robin, with health "
            "checks every 10 seconds. Traffic spikes of up to 10x baseline load are absorbed gracefully "
            "without impacting p99 latency. Rate limiting is applied at the API gateway layer — 1,000 "
            "requests per second per tenant — to prevent any single consumer from monopolizing resources "
            "during peak load scenarios. Cooldown periods of five minutes prevent thrashing after scale-out events."
        ),
    },
    {
        "id": "doc_002",
        "title": "Multi-Tier Caching Architecture",
        "category": "performance",
        "content": (
            "A three-tier caching architecture minimizes database round-trips and reduces backend latency. "
            "The L1 in-process cache uses an LRU (Least Recently Used) eviction policy with a maximum of "
            "500 entries per service instance, providing sub-millisecond lookups for the hottest data. "
            "L2 is a Redis Cluster with six shards and one replica per shard, offering shared cache state "
            "across all service instances. Cache entries carry a configurable TTL (time-to-live), defaulting "
            "to 300 seconds for semi-static data and 30 seconds for volatile metrics. L3 is a CDN edge cache "
            "for static assets and API responses suitable for public consumption. Cache invalidation uses a "
            "publish-subscribe model — producers publish 'invalidation events' on change, and all consumers "
            "flush the relevant keys atomically. Hit rates above 85% at the Redis layer are maintained through "
            "careful key design using consistent hashing, preventing cache stampede via probabilistic early expiry."
        ),
    },
    {
        "id": "doc_003",
        "title": "Fault Tolerance & Circuit Breaker Patterns",
        "category": "reliability",
        "content": (
            "Resilience is built into the service mesh through the circuit breaker pattern implemented via "
            "Resilience4j. Every inter-service HTTP call is wrapped in a circuit breaker with a configurable "
            "failure threshold of 50% over a 10-second sliding window. When the threshold is breached, the "
            "circuit transitions to OPEN state, rejecting calls immediately and returning a cached fallback "
            "response or a graceful degradation payload. After a configurable wait duration (default 60 seconds), "
            "the circuit transitions to HALF-OPEN, allowing a limited number of probe requests through. "
            "Successful probes reset the circuit to CLOSED. Retry policies use exponential backoff with jitter — "
            "base 500ms, maximum 30 seconds — to avoid thundering herd problems. Bulkhead isolation separates "
            "thread pools for critical and non-critical downstream services, ensuring a slowdown in one dependency "
            "does not exhaust the global thread pool. Dead letter queues capture failed async messages for "
            "offline reprocessing without data loss."
        ),
    },
    {
        "id": "doc_004",
        "title": "Database Scaling with Read Replicas & Sharding",
        "category": "data",
        "content": (
            "The persistence layer is built on PostgreSQL 15 with a primary-replica topology to scale read "
            "workloads independently from writes. Three synchronous read replicas handle SELECT queries, "
            "with the application-level connection pool routing reads via PgBouncer. Write operations are "
            "directed exclusively to the primary node, with synchronous replication ensuring zero data loss "
            "on failover. For horizontally sharded workloads exceeding 500GB, Citus Data extension is used "
            "to distribute rows across shards by tenant_id using consistent hashing. Each shard is itself a "
            "primary-replica pair. Connection pooling is managed by PgBouncer in transaction mode, capping "
            "maximum server connections at 200. Database migrations are applied using Flyway in a "
            "backwards-compatible manner, supporting zero-downtime deploys. Slow query logs (> 500ms) are "
            "captured, and automatic VACUUM and ANALYZE jobs run nightly to maintain index performance."
        ),
    },
    {
        "id": "doc_005",
        "title": "Authentication, Authorization & API Security",
        "category": "security",
        "content": (
            "User authentication is handled via OAuth 2.0 Authorization Code flow with PKCE for public clients. "
            "Access tokens are short-lived JWTs (15 minutes) signed with RS256 asymmetric keys, and refresh "
            "tokens are stored as opaque, single-use tokens in a Redis-backed token store with a 30-day TTL. "
            "Role-Based Access Control (RBAC) is enforced at the API gateway using a policy engine (OPA — "
            "Open Policy Agent), which evaluates claims in the JWT against resource-level permissions. "
            "All data in transit is encrypted with TLS 1.3; data at rest uses AES-256-GCM via cloud KMS-managed "
            "keys. API keys for machine-to-machine (M2M) communication are hashed with bcrypt before storage "
            "and validated using constant-time comparison to prevent timing attacks. Security headers "
            "(HSTS, CSP, X-Frame-Options) are injected by the gateway on all responses. Secrets rotation "
            "is automated via Vault Agent, with zero-downtime re-keying supported through dual-key validation windows."
        ),
    },
    {
        "id": "doc_006",
        "title": "Observability: Monitoring, Tracing & Alerting",
        "category": "observability",
        "content": (
            "The observability stack follows the three pillars: metrics, logs, and distributed traces. "
            "Prometheus scrapes metrics from every service every 15 seconds, with Grafana dashboards providing "
            "real-time visualization of RED metrics (Rate, Errors, Duration) per service. Structured logs "
            "(JSON format) are emitted to stdout and collected by a Fluent Bit DaemonSet, forwarded to "
            "Elasticsearch, and visualized in Kibana. Distributed tracing uses OpenTelemetry auto-instrumentation "
            "with Jaeger as the backend, enabling end-to-end request tracing across service boundaries with "
            "a 10% sampling rate (100% for errors). Alerts are configured in Prometheus AlertManager: p99 "
            "latency > 500ms for 5 minutes, error rate > 1% for 2 minutes, and pod restart count > 3 in "
            "10 minutes trigger PagerDuty notifications. SLO burn-rate alerts catch SLA violations before "
            "they fully materialize. Runbooks are linked directly from alert definitions to reduce MTTR."
        ),
    },
    {
        "id": "doc_007",
        "title": "CI/CD Pipeline & Deployment Automation",
        "category": "devops",
        "content": (
            "The continuous integration and delivery pipeline is orchestrated via GitHub Actions with "
            "environment-specific promotion gates. Every pull request triggers a multi-stage pipeline: "
            "static analysis (Ruff, MyPy), unit tests with 80% coverage gate, integration tests against "
            "a Docker Compose test environment, and SAST scanning via Semgrep. On merge to main, the "
            "pipeline builds a multi-arch Docker image (linux/amd64, linux/arm64), pushes to Google "
            "Artifact Registry, and triggers a Helm chart upgrade in the staging namespace. Production "
            "deployments require a manual approval gate and use a blue/green strategy: the new version "
            "is deployed alongside the old, traffic is shifted 10% → 50% → 100% over 20 minutes with "
            "automated rollback if error rate exceeds 0.5%. Infrastructure is provisioned with Terraform, "
            "with state stored in GCS and locking via Cloud Spanner. All secrets are injected at runtime "
            "from Vault, never baked into images."
        ),
    },
    {
        "id": "doc_008",
        "title": "Microservices Communication: gRPC & Service Mesh",
        "category": "architecture",
        "content": (
            "Synchronous inter-service communication uses gRPC over HTTP/2, leveraging Protocol Buffers "
            "for schema-enforced, efficient binary serialization. Service discovery is handled by Kubernetes "
            "DNS with headless services for direct pod addressing in stateful workloads. The Istio service "
            "mesh provides transparent mTLS encryption for all east-west traffic, eliminating the need for "
            "application-level TLS management. Istio's traffic management features enable fine-grained "
            "routing: canary deployments route 5% of traffic to the new version based on request headers, "
            "and traffic mirroring shadows production traffic to staging for validation. gRPC streaming "
            "is used for real-time event feeds (e.g., live order status updates), with client-side load "
            "balancing using the round_robin policy across endpoints. Deadlines are propagated through the "
            "entire call chain using gRPC context, ensuring upstream timeouts automatically cancel downstream work."
        ),
    },
    {
        "id": "doc_009",
        "title": "API Gateway: Rate Limiting, Routing & Throttling",
        "category": "infrastructure",
        "content": (
            "Kong API Gateway serves as the single ingress point for all external traffic, running in "
            "DB-less declarative mode for GitOps-friendly configuration management. Rate limiting is "
            "implemented using the sliding window algorithm, enforced at both the consumer and route "
            "levels: 1,000 req/min per API key globally, with burst allowance of 200 requests over "
            "any 10-second window using the token bucket algorithm. Request throttling beyond limits "
            "returns HTTP 429 with Retry-After headers. The gateway handles JWT validation, stripping "
            "auth headers before forwarding to upstream services. Request/response transformation plugins "
            "normalize API versions, add correlation IDs to every request, and enforce request size limits "
            "(max 10MB payload). Upstream load balancing at the gateway uses least-connections algorithm "
            "with passive health checks. Edge-level CORS policies, bot detection (via fingerprinting), "
            "and IP allowlisting for sensitive admin endpoints are all managed centrally at the gateway layer."
        ),
    },
    {
        "id": "doc_010",
        "title": "Asynchronous Processing with Message Queues",
        "category": "messaging",
        "content": (
            "Asynchronous workloads are decoupled from the synchronous request path using RabbitMQ as "
            "the message broker, configured in a highly available cluster with three nodes and quorum queues "
            "for durability. Producers publish messages to topic exchanges with routing keys that fan out "
            "to multiple consumer queues. Each consumer service processes messages with at-least-once "
            "delivery semantics, using idempotency keys to deduplicate reprocessed messages. Message "
            "schemas are validated against a central Schema Registry using Avro, preventing incompatible "
            "producers from publishing malformed events. Dead letter exchanges (DLX) capture messages "
            "that fail processing after three retries, with exponential backoff between attempts (2s, 8s, 32s). "
            "Operations teams receive alerts when DLX queue depth exceeds 100 messages. Priority queues "
            "with three tiers (high, normal, low) ensure critical transactional messages are always "
            "processed ahead of background analytics workloads. Consumer prefetch count is tuned to "
            "match downstream service throughput to avoid queue buildup."
        ),
    },
]
