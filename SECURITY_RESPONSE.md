# 🚨 SECURITY INCIDENT RESPONSE

## Critical: Production Secrets Exposed in Git Repository

**Date:** 2025-11-25  
**Severity:** CRITICAL  
**Status:** REQUIRES IMMEDIATE ACTION

## Summary

The `.env` file containing production secrets has been committed to the git repository. The following sensitive credentials are exposed:

- **Django SECRET_KEY**: Used for cryptographic signing, session security, and CSRF protection
- **Database Password**: PostgreSQL database credentials
- **TELEGRAM_TOKEN**: Telegram bot authentication token
- **OPENAI_API_KEY**: OpenAI API authentication key

Additionally, IDE configuration files (`.idea/`) and log files have been committed.

## Immediate Actions Required

### 1. Rotate All Exposed Credentials (DO THIS FIRST)

#### A. Django SECRET_KEY
1. Generate a new secret key:
   ```bash
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```
2. Update `.env` file with the new `SECRET_KEY`
3. **Important:** All existing user sessions will be invalidated. Users will need to log in again.

#### B. Database Password
1. Connect to PostgreSQL as superuser
2. Change the password for `englishbot_user`:
   ```sql
   ALTER USER englishbot_user WITH PASSWORD 'new_secure_password';
   ```
3. Update `.env` file with the new `DB_PASSWORD`
4. Restart the application

#### C. Telegram Bot Token
1. Go to [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/mybots` and select your bot
3. Choose "API Token" → "Revoke current token"
4. Generate a new token
5. Update `.env` file with the new `TELEGRAM_TOKEN`
6. Restart the bot

#### D. OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Revoke the exposed API key
3. Generate a new API key
4. Update `.env` file with the new `OPENAI_API_KEY`
5. Restart the application

### 2. Remove Files from Git History

The files have been removed from git tracking, but if they were already committed to history, you need to clean the history:

#### Option A: If files were only staged (not yet pushed)
```bash
# Files are already removed from tracking with git rm --cached
# Just commit the removal:
git commit -m "Remove sensitive files from version control"
```

#### Option B: If files were committed and pushed to remote
**WARNING:** This rewrites git history. Coordinate with your team first.

```bash
# Remove from all commits in history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env .idea/ logs/errors.log logs/errors.log.*" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (coordinate with team first!)
git push origin --force --all
git push origin --force --tags
```

**Alternative (safer):** Use `git-filter-repo` tool:
```bash
pip install git-filter-repo
git filter-repo --path .env --path .idea/ --path logs/ --invert-paths
```

### 3. Verify .gitignore is Working

The `.gitignore` file has been created/updated to exclude:
- `.env` and variants
- `.idea/` directory
- `logs/` directory
- Other sensitive/temporary files

Verify it's working:
```bash
git status
# Should NOT show .env, .idea/, or logs/ files
```

### 4. Audit Access

1. Review who has access to the repository
2. If the repository is public or shared, assume all secrets are compromised
3. Check repository access logs if available
4. Review any forks or clones of the repository

### 5. Monitor for Unauthorized Access

After rotating credentials, monitor for:
- Unauthorized database access
- Unauthorized Telegram bot usage
- Unauthorized OpenAI API usage
- Unusual application behavior

## Prevention

1. **Never commit `.env` files** - Always use `.gitignore`
2. **Use environment-specific configs** - Different `.env` files for dev/staging/prod
3. **Use secret management** - Consider using:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Django's `python-decouple` with separate config files
4. **Pre-commit hooks** - Set up hooks to prevent committing secrets:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
5. **Regular audits** - Periodically check for committed secrets using tools like:
   - `git-secrets`
   - `truffleHog`
   - GitHub's secret scanning

## Checklist

- [ ] Rotate Django SECRET_KEY
- [ ] Rotate database password
- [ ] Rotate Telegram bot token
- [ ] Rotate OpenAI API key
- [ ] Remove files from git tracking (completed)
- [ ] Clean git history if needed
- [ ] Verify .gitignore is working
- [ ] Update all deployment environments with new credentials
- [ ] Restart all services
- [ ] Monitor for unauthorized access
- [ ] Document incident and lessons learned

## Notes

- All credentials in the exposed `.env` file are considered compromised
- Even if the repository is private, treat all secrets as public
- The `.gitignore` file will prevent future commits of these files
- Consider implementing secret scanning in CI/CD pipelines

---

**Last Updated:** 2025-11-25

