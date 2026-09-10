# docker/ssl/README.md

## SSL Certificates

This directory holds TLS certificates used by Nginx.

### Development (self-signed)

Run the generation script:
```bash
chmod +x docker/ssl/gen_dev_certs.sh
./docker/ssl/gen_dev_certs.sh
```

This creates `cert.pem` and `key.pem` valid for 10 years. Browsers will show
a security warning for self-signed certs — this is expected in development.

### Production (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot

# Obtain certificate (domain must point to your server)
certbot certonly --standalone -d yourdomain.com

# Copy to this directory
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem   key.pem
chmod 600 key.pem
```

Set up auto-renewal:
```bash
# Add to crontab
0 3 * * * certbot renew --quiet && docker compose exec nginx nginx -s reload
```

### Files in this directory

| File | Purpose |
|------|---------|
| `cert.pem` | TLS certificate (public — safe to commit in dev) |
| `key.pem`  | Private key (**NEVER commit to git**) |
| `gen_dev_certs.sh` | Script to generate dev self-signed certs |

> `*.pem` files are in `.gitignore` — never commit private keys.