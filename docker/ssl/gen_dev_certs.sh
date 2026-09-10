#!/usr/bin/env bash
# docker/ssl/gen_dev_certs.sh
# ============================
# Generate self-signed SSL certificates for LOCAL DEVELOPMENT ONLY.
# For production, use Let's Encrypt (certbot) or a real CA.
#
# USAGE:
#   chmod +x docker/ssl/gen_dev_certs.sh
#   ./docker/ssl/gen_dev_certs.sh
#
# OUTPUT:
#   docker/ssl/cert.pem  — self-signed certificate (10 year validity)
#   docker/ssl/key.pem   — private key (RSA 2048-bit)
#
# PRODUCTION:
#   Replace these files with real certificates from Let's Encrypt:
#   certbot certonly --standalone -d yourdomain.com
#   cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem docker/ssl/cert.pem
#   cp /etc/letsencrypt/live/yourdomain.com/privkey.pem   docker/ssl/key.pem

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_FILE="${SCRIPT_DIR}/cert.pem"
KEY_FILE="${SCRIPT_DIR}/key.pem"

echo "Generating self-signed SSL certificate for development..."

openssl req -x509 -nodes -days 3650 \
    -newkey rsa:2048 \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -subj "/C=US/ST=Development/L=Local/O=LLMPlatform/OU=Dev/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:llm-api,IP:127.0.0.1"

chmod 600 "${KEY_FILE}"
chmod 644 "${CERT_FILE}"

echo "✅ Certificate generated:"
echo "   cert.pem: ${CERT_FILE}"
echo "   key.pem:  ${KEY_FILE}"
echo ""
echo "Certificate details:"
openssl x509 -in "${CERT_FILE}" -noout -subject -dates
echo ""
echo "⚠️  These are DEVELOPMENT-ONLY self-signed certificates."
echo "   Browsers will show a security warning — this is expected."
echo "   Use real certificates from Let's Encrypt in production."