# 🛡️ SECURITY
ParkCircus Productions

## 📋 Policy Overview

This project follows a Risk-Based Security Model. We prioritize vulnerabilities that impact our edge-computing cluster and VLM benchmarking integrity.

Version | Status
---|---
Current (1.0.x) | ✅ Supported
Legacy (< 1.0) | ❌ Unsupported

---

## 🔍 Current Security Posture

We monitor several dozen nodes (primarily single board microcomputers) using a layered defense strategy.

* **Dependency Auditing**: `Safety CLI`
* **Malware Protection**: `ClamAV` (Staggered Cluster Scans)
* **Reporting**: Automated Heartbeat Dashboard via LAN web server

---

## 🛠️ Remediation Status

### ✅ FIXED: Critical Infrastructure Risks

These vulnerabilities were patched to prevent unauthorized access to the SOHO LAN.

* **🚀 RCE**: `distributed` → Upgraded to `2026.1.2` (CVE-2026-23528)
* **🤖 AI Platform**: `google-cloud-aiplatform` → Upgraded to `1.139.0` (CVE-2026-2473)
* **🎥 Injection**: `yt-dlp` → Upgraded to `2026.2.21` (Command Injection risk)

### ⏳ DEFERRED: Contextual Exceptions

The following are cataloged but not remediated, as they do not impact the vlm_learn runtime environment.

* **🔑 Authlib (84339)**: CSRF risk. **Reason**: Library is used for CLI-only auth; no web-facing cache is active.
* **📊 marshmallow (CVE-2025-68480)**: DoS risk. **Reason** Data input is restricted to trusted local datasets/models.

---

## 🚨 Reporting a Vulnerability

If you find a security flaw, please do not open a public Issue.

* **📩 Email**: security@parkcircus.org
* **🕒 Response Time**: Acknowledgment within 48 Hours.
* **🛠️ Fix Timeline**: Target resolution within 10 Business Days. within 48 hours and provide a fix or mitigation plan within 10 business days.
