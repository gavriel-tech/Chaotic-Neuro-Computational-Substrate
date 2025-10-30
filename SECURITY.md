# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. **Do Not** Create a Public Issue

Please do not create a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Report Privately

Send a detailed report via: **https://gavriel.tech** (use contact form) or create a private security advisory on GitHub

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 1 week
- **Fix Timeline**: Varies by severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: Best effort

### 4. Disclosure Policy

- We will acknowledge your report
- We will validate and investigate
- We will develop and test a fix
- We will release a security advisory
- We will credit you (unless you prefer to remain anonymous)

## Security Best Practices

### Deployment

1. **Never expose to the public internet without authentication**
   ```bash
   # Bad: Accessible to anyone
   python -m src.main --host 0.0.0.0 --port 8000
   
   # Good: Behind authentication/firewall
   # Use reverse proxy with authentication
   ```

2. **Enable rate limiting in production**
   ```python
   # In src/api/server.py
   RATE_LIMIT_ENABLED = True
   RATE_LIMIT_REQUESTS_PER_MINUTE = 60
   ```

3. **Use environment variables for secrets**
   ```bash
   # Never commit secrets to git
   export GMCS_API_KEY=your-secret-key
   export GMCS_SECRET_KEY=$(openssl rand -hex 32)
   ```

4. **Keep dependencies updated**
   ```bash
   pip list --outdated
   pip install --upgrade package-name
   ```

5. **Run with minimal privileges**
   ```bash
   # Don't run as root
   useradd -m gmcs
   su - gmcs
   python -m src.main
   ```

### API Security

1. **CORS Configuration**
   ```python
   # Restrict CORS in production
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://gavriel.tech"],  # Not ["*"]
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["Content-Type", "Authorization"],
   )
   ```

2. **Input Validation**
   - All user inputs are validated via Pydantic
   - Parameter ranges are enforced
   - File uploads should be scanned

3. **WebSocket Security**
   - Implement connection authentication
   - Limit concurrent connections
   - Monitor for abuse patterns

### Known Security Considerations

#### 1. GPU Memory Exhaustion

**Risk**: Malicious users could request maximum nodes/grid sizes

**Mitigation**:
```python
# Enforce resource limits
MAX_NODES_PER_SESSION = 1024
MAX_GRID_SIZE = 512
MAX_CONCURRENT_SESSIONS = 10
```

#### 2. Denial of Service via Parameter Abuse

**Risk**: Extreme parameter values could cause instability

**Mitigation**:
- Parameter validation with strict ranges
- CFL stability checks
- Automatic state validation

#### 3. WebSocket Message Flooding

**Risk**: Clients could flood the server with messages

**Mitigation**:
- Rate limiting (implemented)
- Message size limits
- Connection timeout

#### 4. Dependency Vulnerabilities

**Risk**: Third-party packages may have vulnerabilities

**Mitigation**:
```bash
# Regular security audits
pip install safety
safety check

# Or use pip-audit
pip install pip-audit
pip-audit
```

### Docker Security

1. **Don't run as root**
   ```dockerfile
   # Add to Dockerfile
   RUN useradd -m gmcs
   USER gmcs
   ```

2. **Use official base images**
   ```dockerfile
   FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
   # Verify image signatures
   ```

3. **Scan for vulnerabilities**
   ```bash
   docker scan gmcs-backend:latest
   ```

4. **Limit container resources**
   ```yaml
   # docker-compose.yml
   services:
     gmcs-backend:
       deploy:
         resources:
           limits:
             cpus: '4'
             memory: 16G
   ```

## Security Checklist for Production

- [ ] Change default secret keys
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Use HTTPS/WSS (not HTTP/WS)
- [ ] Set up authentication
- [ ] Enable logging and monitoring
- [ ] Regular dependency updates
- [ ] Firewall configuration
- [ ] Backup strategy
- [ ] Incident response plan

## Contact

For security concerns, contact:
- Website: https://gavriel.tech
- GitHub Security Advisories: https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate/security/advisories

For general questions, use GitHub Discussions.

---

**Last Updated**: January 2025


