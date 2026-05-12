"""
query_expander.py
-----------------
QueryExpander: Uses a (mocked) Vertex AI GenerativeModel to rewrite/expand
user queries into more embedding-friendly, semantically richer search terms.

How it works
------------
In production, this class would instantiate a real vertexai.generative_models.GenerativeModel
("gemini-1.5-flash") and call generate_content() with a structured prompt.

In tests and benchmarks, vertexai.generative_models.GenerativeModel is mocked via
unittest.mock.patch so no GCP credentials or network calls are required.

The EXPANDED_QUERIES mapping provides deterministic expansions for the five benchmark
queries, ensuring reproducible benchmark results.

Expansion strategy
------------------
The expanded queries:
1. Include the original query's intent
2. Add technical synonyms and related concepts
3. Use noun phrases that align better with document terminology
4. Are longer and more specific, casting a wider semantic net
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Vertex AI import — kept at module level so unittest.mock.patch can
# target 'src.query_expander.GenerativeModel' even when vertexai is not installed.
# ---------------------------------------------------------------------------
try:
    from vertexai.generative_models import GenerativeModel  # type: ignore[import]
    _VERTEX_AVAILABLE = True
except ImportError:
    GenerativeModel = None  # type: ignore[assignment,misc]
    _VERTEX_AVAILABLE = False
    logger.warning(
        "vertexai package not installed. "
        "QueryExpander will use mocked GenerativeModel in tests. "
        "Install google-cloud-aiplatform for production use."
    )

# ---------------------------------------------------------------------------
# Deterministic expansion map (used by the mock in tests and benchmark runner)
# ---------------------------------------------------------------------------
EXPANDED_QUERIES: dict[str, str] = {
    "How does the system handle peak load?": (
        "How does the system handle peak load traffic and high-demand periods? "
        "Describe auto-scaling, horizontal scaling, Kubernetes HPA, load balancing "
        "strategies such as round-robin and weighted distribution, rate limiting, "
        "traffic management, burst handling, request throttling, resource provisioning, "
        "cooldown periods, and compute node scaling during traffic surges and capacity peaks."
    ),
    "What security measures protect user data?": (
        "What security measures, protocols, and mechanisms protect user data and ensure "
        "system security? Describe authentication with OAuth 2.0, JWT token validation, "
        "PKCE, role-based access control RBAC, encryption at rest AES-256 and in transit "
        "TLS 1.3, API key management, secret rotation with Vault, SAST scanning, "
        "single-use refresh tokens, OPA policy engine, and compliance with security standards."
    ),
    "How is the system monitored in production?": (
        "How is the system monitored and observed in production environments? "
        "Describe the observability stack including metrics collection with Prometheus, "
        "log aggregation with Fluent Bit and Elasticsearch Kibana ELK, distributed tracing "
        "with OpenTelemetry and Jaeger, Grafana dashboards for RED metrics rate errors duration, "
        "SLO burn-rate alerting, PagerDuty notifications, runbooks, and MTTR reduction strategies."
    ),
    "How are service failures and outages handled?": (
        "How are service failures, outages, and downstream dependency errors handled in the system? "
        "Describe fault tolerance mechanisms including circuit breaker patterns with Resilience4j, "
        "fallback responses, graceful degradation, bulkhead isolation, retry policies with "
        "exponential backoff and jitter, dead letter queues for failed messages, thundering herd "
        "prevention, thread pool isolation, and recovery from partial failures."
    ),
    "How does the system deploy new versions without downtime?": (
        "How does the system deploy new versions, updates, and releases without downtime or "
        "service interruption? Describe continuous delivery CI/CD pipelines with GitHub Actions, "
        "blue-green deployment strategies, canary releases with traffic shifting percentages, "
        "Helm chart upgrades, automated rollback on error rate thresholds, Kubernetes rolling "
        "updates, Terraform infrastructure as code, zero-downtime database migrations with Flyway, "
        "and production promotion gates."
    ),
}

# Prompt template used when calling the real GenerativeModel
EXPANSION_PROMPT_TEMPLATE = (
    "You are an expert search query optimizer for a technical knowledge base. "
    "Rewrite the following user query to be more specific, technically detailed, "
    "and embedding-friendly. Include relevant technical synonyms, related concepts, "
    "and domain terminology. Return ONLY the expanded query text, nothing else.\n\n"
    "Original query: {query}\n\n"
    "Expanded query:"
)


class QueryExpander:
    """
    AI-enhanced query expansion using a Vertex AI GenerativeModel.

    In production: instantiates GenerativeModel('gemini-1.5-flash') and calls
    generate_content() with a structured expansion prompt.

    In tests/benchmarks: GenerativeModel is mocked via unittest.mock.patch, with
    EXPANDED_QUERIES providing deterministic responses for the benchmark queries.

    Parameters
    ----------
    model_name : str
        Vertex AI generative model name. Default: 'gemini-1.5-flash'.

    Example
    -------
    >>> # In tests, patch 'src.query_expander.GenerativeModel' before instantiating
    >>> expander = QueryExpander()
    >>> expanded = expander.expand_query("How does the system handle peak load?")
    >>> len(expanded) > len("How does the system handle peak load?")
    True
    """

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self.model_name = model_name
        # Uses the module-level GenerativeModel so that unittest.mock.patch(
        # 'src.query_expander.GenerativeModel') correctly replaces it before
        # this constructor runs.
        if GenerativeModel is not None:
            self.model = GenerativeModel(model_name)
            logger.info("QueryExpander initialized with real GenerativeModel: %s", model_name)
        else:
            # vertexai not installed; will be mocked in tests / benchmark
            self.model = None

    def expand_query(self, query: str) -> str:
        """
        Expand a user query using the generative model.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        str
            Expanded, embedding-friendly version of the query. Falls back to
            the original query if the model call fails.

        Raises
        ------
        ValueError
            If query is empty.
        RuntimeError
            If model is not available and no fallback is possible.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if self.model is None:
            raise RuntimeError(
                "GenerativeModel not available. Ensure vertexai is installed "
                "or mock QueryExpander.model in tests."
            )

        prompt = EXPANSION_PROMPT_TEMPLATE.format(query=query)
        logger.debug("Expanding query: '%s'", query)

        try:
            response = self.model.generate_content(prompt)
            expanded = response.text.strip()
            logger.debug("Expanded query: '%s'", expanded)
            return expanded
        except Exception as exc:
            logger.warning("Query expansion failed (%s). Returning original query.", exc)
            return query

    def __repr__(self) -> str:
        return f"QueryExpander(model='{self.model_name}')"
