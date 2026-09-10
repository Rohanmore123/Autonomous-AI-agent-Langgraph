"""
scripts/load_test.py
=====================
Locust load test for the LLM Platform.

SIMULATES REALISTIC USER BEHAVIOUR:
  • Login → get JWT
  • Browse conversations
  • Send chat messages (RAG + direct)
  • Search documents
  • Check task list

USAGE:
  # Install locust: pip install locust
  # Headless (CI/CD):
  locust -f scripts/load_test.py \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --host http://localhost:8000 \
    --html report.html

  # Interactive UI (browser):
  locust -f scripts/load_test.py --host http://localhost:8000
  # Open http://localhost:8089

TARGET METRICS FOR 10K CONCURRENT USERS (with 4+ API servers):
  RPS:              5,000–10,000 req/s
  P50 latency:      < 500ms
  P95 latency:      < 3,000ms
  P99 latency:      < 10,000ms (LLM calls are inherently slow)
  Error rate:       < 0.1%
  Cache hit rate:   > 40%

INTERPRETATION:
  • High P99 on /chat is expected (LLM calls = 2–10s)
  • High P50 on /documents/search should be < 200ms (Weaviate is fast)
  • Error spikes on /chat may indicate circuit breaker opening
"""

import json
import random
from typing import Optional

from locust import HttpUser, SequentialTaskSet, between, task


# ---------------------------------------------------------------------------
# Realistic test data
# ---------------------------------------------------------------------------

CHAT_MESSAGES = [
    "What are the main findings in the uploaded report?",
    "Summarise the key points from the documentation.",
    "What does the document say about performance?",
    "Find information about the configuration options.",
    "Can you explain the architecture described in the docs?",
    "What are the security recommendations?",
    "How do I set up the system according to the guide?",
    "What are the known limitations mentioned?",
    "List the supported integrations.",
    "What version was the API updated in?",
]

DOCUMENT_QUERIES = [
    "performance optimisation",
    "security configuration",
    "API authentication",
    "database setup",
    "deployment guide",
    "rate limiting",
    "error handling",
    "monitoring setup",
]

TEST_USERS = [
    {"email": f"loadtest{i}@example.com", "password": "TestUser@123456"}
    for i in range(1, 101)
]


# ---------------------------------------------------------------------------
# Task sets
# ---------------------------------------------------------------------------

class AuthenticatedUserBehavior(SequentialTaskSet):
    """
    Simulates a realistic user session:
    1. Register/Login
    2. Browse conversations
    3. Send chat messages
    4. Search documents
    5. Check tasks
    """

    access_token: Optional[str] = None
    conversation_id: Optional[str] = None

    def on_start(self) -> None:
        """Login at the start of each user session."""
        user = random.choice(TEST_USERS)
        self._login(user["email"], user["password"])

    def _login(self, email: str, password: str) -> None:
        """Attempt login; register if user doesn't exist."""
        resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
            name="POST /auth/login",
        )
        if resp.status_code == 401:
            # User doesn't exist — register first
            reg_resp = self.client.post(
                "/api/v1/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "username": email.split("@")[0].replace(".", "_"),
                },
                name="POST /auth/register",
            )
            if reg_resp.status_code == 201:
                resp = self.client.post(
                    "/api/v1/auth/login",
                    json={"email": email, "password": password},
                    name="POST /auth/login",
                )

        if resp.status_code == 200:
            data = resp.json()
            self.access_token = data.get("data", {}).get("access_token")
        else:
            self.access_token = None

    def _headers(self) -> dict:
        if self.access_token:
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}

    @task(1)
    def get_profile(self) -> None:
        """GET /auth/me — lightweight, frequent."""
        self.client.get("/api/v1/auth/me", headers=self._headers(), name="GET /auth/me")

    @task(2)
    def list_conversations(self) -> None:
        """GET /chat/conversations — medium frequency."""
        resp = self.client.get(
            "/api/v1/chat/conversations",
            headers=self._headers(),
            name="GET /chat/conversations",
        )
        if resp.status_code == 200:
            items = resp.json().get("data", {}).get("items", [])
            if items:
                self.conversation_id = items[0]["id"]

    @task(5)
    def send_chat_message(self) -> None:
        """POST /chat — the most common and expensive operation."""
        message = random.choice(CHAT_MESSAGES)
        payload = {
            "message": message,
            "agent_type": random.choice(["rag", "direct", "router"]),
            "include_sources": random.random() > 0.7,
        }
        if self.conversation_id and random.random() > 0.5:
            payload["conversation_id"] = self.conversation_id

        with self.client.post(
            "/api/v1/chat",
            json=payload,
            headers=self._headers(),
            name="POST /chat",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data", {}).get("conversation_id"):
                    self.conversation_id = data["data"]["conversation_id"]
                resp.success()
            elif resp.status_code == 429:
                resp.failure(f"Rate limited: {resp.text[:100]}")
            elif resp.status_code >= 500:
                resp.failure(f"Server error: {resp.status_code}")
            else:
                resp.success()  # 4xx are expected (bad requests)

    @task(3)
    def search_documents(self) -> None:
        """POST /documents/search — moderate frequency."""
        query = random.choice(DOCUMENT_QUERIES)
        self.client.post(
            "/api/v1/documents/search",
            json={"query": query, "top_k": 5, "alpha": 0.5},
            headers=self._headers(),
            name="POST /documents/search",
        )

    @task(2)
    def list_documents(self) -> None:
        """GET /documents — lightweight listing."""
        self.client.get(
            "/api/v1/documents",
            headers=self._headers(),
            name="GET /documents",
        )

    @task(1)
    def health_check(self) -> None:
        """GET /health — simulates load balancer health probes."""
        self.client.get("/health", name="GET /health")


# ---------------------------------------------------------------------------
# User classes (different load patterns)
# ---------------------------------------------------------------------------

class RegularUser(HttpUser):
    """
    Standard user — sends chat messages, searches documents.
    Wait time: 1–5 seconds between tasks (simulates reading time).
    """
    tasks = [AuthenticatedUserBehavior]
    wait_time = between(1, 5)
    weight = 8   # 80% of virtual users are regular users


class PowerUser(HttpUser):
    """
    Power user — heavy chat usage, rapid-fire requests.
    Simulates enterprise users with heavy automation.
    """
    tasks = [AuthenticatedUserBehavior]
    wait_time = between(0.1, 1)
    weight = 2   # 20% are power users


# ---------------------------------------------------------------------------
# Standalone runner (for quick testing without locust UI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("""
Run with locust:
  locust -f scripts/load_test.py --headless --users 100 --spawn-rate 10 --run-time 2m --host http://localhost:8000

Or start the UI:
  locust -f scripts/load_test.py --host http://localhost:8000
  Open: http://localhost:8089
    """)