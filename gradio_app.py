"""Gradio interface for the LLM Agentic Platform API.

Run locally:
    uv run --extra ui python gradio_app.py

The API base URL can be overridden with API_BASE_URL.
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", "120"))


def _request(
    method: str,
    path: str,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{path}",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Cannot reach the API at {API_BASE_URL}. Start the FastAPI server first."
        ) from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"API returned an invalid response ({response.status_code}).") from exc

    if response.is_error:
        detail = body.get("detail") or body.get("message") or body.get("errors")
        raise RuntimeError(f"API error {response.status_code}: {detail or 'Request failed'}")

    return body


def register(
    email: str,
    username: str,
    password: str,
    full_name: str,
) -> str:
    if not email or not username or not password:
        return "Enter email, username, and password."

    try:
        _request(
            "POST",
            "/api/v1/auth/register",
            payload={
                "email": email,
                "username": username,
                "password": password,
                "full_name": full_name or None,
            },
        )
        return "Account created. Use the Login tab to sign in."
    except RuntimeError as exc:
        if "Email already registered" in str(exc):
            return "This email is already registered. Please use the Login tab."
        return str(exc)


def login(email: str, password: str) -> tuple[str, str, str]:
    if not email or not password:
        return "", "", "Enter your email and password."

    try:
        body = _request(
            "POST",
            "/api/v1/auth/login",
            payload={"email": email, "password": password},
        )
        token_data = body.get("data") or {}
        access_token = token_data.get("access_token", "")
        refresh_token = token_data.get("refresh_token", "")
        if not access_token:
            return "", "", "Login succeeded but no access token was returned."
        return access_token, refresh_token, "Signed in successfully."
    except RuntimeError as exc:
        return "", "", str(exc)


def logout():
    return "", "", "", "Signed out.", []


def send_message(message, history, agent_type, token, conversation_id):
    if not token:
        return history, "Sign in before sending messages.", conversation_id
    if not message.strip():
        return history, "Enter a message.", conversation_id

    try:
        body = _request(
            "POST",
            "/api/v1/chat",
            token=token,
            payload={
                "conversation_id": conversation_id or None,
                "message": message.strip(),
                "agent_type": agent_type,
                "include_sources": agent_type == "rag",
            },
        )
        response_data = body.get("data") or {}
        answer = response_data.get("content", "The API returned no response content.")
        conversation_id = response_data.get("conversation_id", conversation_id)
        history = history or []
        history.append((message, answer))
        return history, f"Agent: {agent_type}", conversation_id
    except RuntimeError as exc:
        if "Google account not connected" in str(exc):
            return (
                history or [],
                "Google is not connected. Click Connect Google, complete authorization, and try again.",
                conversation_id,
            )
        return history or [], str(exc), conversation_id


def current_user(token: str) -> str:
    if not token:
        return "Not signed in"
    try:
        body = _request("GET", "/api/v1/auth/me", token=token)
        user = body.get("data") or {}
        return f"Signed in as {user.get('email', 'user')}"
    except RuntimeError as exc:
        return str(exc)


def connect_google(token: str) -> str:
    if not token:
        return "Sign in before connecting Google."

    try:
        response = httpx.get(
            f"{API_BASE_URL}/api/v1/auth/google",
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
            follow_redirects=False,
        )
        if response.status_code not in (301, 302, 303, 307, 308):
            try:
                detail = response.json().get("detail", "Unable to start Google authorization")
            except ValueError:
                detail = "Unable to start Google authorization"
            return f"Google authorization failed ({response.status_code}): {detail}"
        location = response.headers.get("location")
        if not location:
            return "Google authorization did not return a redirect URL."
        return f"[Open Google authorization]({location})"
    except httpx.RequestError:
        return f"Cannot reach the API at {API_BASE_URL}."


def send_email(to: str, subject: str, body: str, token: str) -> str:
    if not token:
        return "Sign in before sending email."
    recipients = [address.strip() for address in to.split(",") if address.strip()]
    if not recipients or not subject.strip() or not body.strip():
        return "Enter recipient, subject, and message body."

    try:
        response = _request(
            "POST",
            "/api/v1/gmail/send",
            token=token,
            payload={"to": recipients, "subject": subject.strip(), "body": body.strip()},
        )
        message_id = (response.get("data") or {}).get("message_id", "")
        return f"Email sent successfully. Message ID: {message_id}"
    except RuntimeError as exc:
        if "Google account not connected" in str(exc):
            return "Google is not connected. Click Connect Google first."
        return str(exc)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LLM Agentic Platform") as demo:
        access_token = gr.State("")
        refresh_token = gr.State("")
        conversation_id = gr.State("")

        gr.Markdown("# LLM Agentic Platform")
        gr.Markdown(f"API: `{API_BASE_URL}`")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                status = gr.Textbox(label="Session", value="Not signed in", interactive=False)

                with gr.Tab("Login"):
                    login_email = gr.Textbox(label="Email", placeholder="you@example.com")
                    login_password = gr.Textbox(label="Password", type="password")
                    login_button = gr.Button("Sign in", variant="primary")
                    login_message = gr.Markdown()

                with gr.Tab("Register"):
                    register_email = gr.Textbox(label="Email")
                    register_username = gr.Textbox(label="Username")
                    register_full_name = gr.Textbox(label="Full name")
                    register_password = gr.Textbox(label="Password", type="password")
                    register_button = gr.Button("Create account")
                    register_message = gr.Markdown()

                logout_button = gr.Button("Sign out")
                google_button = gr.Button("Connect Google")
                google_link = gr.Markdown()

                with gr.Accordion("Compose email", open=False):
                    email_to = gr.Textbox(label="To", placeholder="recipient@example.com")
                    email_subject = gr.Textbox(label="Subject")
                    email_body = gr.Textbox(label="Body", lines=5)
                    send_email_button = gr.Button("Send email", variant="primary")
                    email_status = gr.Markdown()

            with gr.Column(scale=2):
                agent_type = gr.Dropdown(
                    choices=["router", "rag", "task", "gmail"],
                    value="router",
                    label="Agent",
                )
                chatbot = gr.Chatbot(label="Conversation", height=520)
                message = gr.Textbox(
                    label="Message",
                    placeholder="Ask a question or tell an agent what to do...",
                    lines=3,
                )
                send_button = gr.Button("Send", variant="primary")
                chat_status = gr.Markdown()

        login_button.click(
            login,
            inputs=[login_email, login_password],
            outputs=[access_token, refresh_token, login_message],
            api_name=False,
        ).then(current_user, inputs=[access_token], outputs=[status], api_name=False)

        register_button.click(
            register,
            inputs=[register_email, register_username, register_password, register_full_name],
            outputs=[register_message],
            api_name=False,
        )

        logout_button.click(
            logout,
            outputs=[access_token, refresh_token, conversation_id, status, chatbot],
            api_name=False,
        )

        google_button.click(
            connect_google,
            inputs=[access_token],
            outputs=[google_link],
            api_name=False,
        )

        send_email_button.click(
            send_email,
            inputs=[email_to, email_subject, email_body, access_token],
            outputs=[email_status],
            api_name=False,
        )

        send_button.click(
            send_message,
            inputs=[message, chatbot, agent_type, access_token, conversation_id],
            outputs=[chatbot, chat_status, conversation_id],
            api_name=False,
        ).then(lambda: "", outputs=[message], api_name=False)
        message.submit(
            send_message,
            inputs=[message, chatbot, agent_type, access_token, conversation_id],
            outputs=[chatbot, chat_status, conversation_id],
            api_name=False,
        ).then(lambda: "", outputs=[message], api_name=False)

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
        show_error=True,
        # share=True
    )
