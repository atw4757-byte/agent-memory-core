# Enterprise

**Pricing anchors at $3,000/month** and scales with load, seats, and compliance requirements. For teams who need private deployment, regulatory controls, and dedicated support.

Contact: **enterprise@divergencerouter.com**

---

## Who This Is For

- **Regulated industries** — healthcare (HIPAA), finance (SOC2 Type II, PCI), government (FedRAMP-adjacent)
- **High-volume production** — 10M+ chunks stored, 1K+ eval runs per month
- **Privacy-sensitive** — customer data that cannot leave your VPC under any circumstance
- **Multi-tenant platforms** — you're embedding agent memory into your own product
- **Teams > 25 engineers** — when the Business tier's 10-seat cap stops fitting

---

## What Enterprise Includes

Everything in [Business](PRICING.md), plus:

### Deployment
- **VPC / private cloud deployment** — observability + eval backend runs inside your AWS/GCP/Azure environment
- **On-premise option** — air-gapped environments supported for government and defense
- **Kubernetes manifests + Terraform** — infrastructure-as-code, no ClickOps
- **BYO-LLM** — bring your own Ollama/vLLM cluster for consolidation, or use our managed workers

### Identity + access
- **SSO / SAML 2.0** — Okta, Auth0, Azure AD, Google Workspace
- **SCIM 2.0 provisioning** — automate user lifecycle from your IdP
- **Role-based access** — admin, workspace owner, editor, viewer, auditor roles out of the box
- **Custom role definitions** — fine-grained permission grants

### Compliance
- **Signed DPA** (Data Processing Addendum) — standard + customizable terms
- **Signed BAA** (Business Associate Agreement) — HIPAA workloads
- **SOC2 Type II** — report available under NDA; Type I on Business tier
- **PCI DSS compliance** — for credential and financial type chunks
- **GDPR / CCPA** — deletion SLA, data residency options
- **Audit logs** — every chunk add/delete/search, with configurable retention (default 2 years)

### Security
- **Customer-managed encryption keys** (CMEK) — you hold the keys; rotate on your schedule
- **Private network only** — no public internet egress
- **IP allowlisting** — API access restricted to your ranges
- **Hardware security module** (HSM) support — for credential-type chunks
- **Penetration testing reports** — annual 3rd-party report available

### Support + SLA
- **99.95% uptime SLA** with service credits
- **24/7 dedicated support** via shared Slack + on-call phone
- **Named technical account manager** (TAM)
- **Quarterly architecture reviews** — we help you optimize memory patterns as you scale
- **Priority feature requests** — your roadmap asks get weight

### Custom work
- **Custom retention policies** — per namespace, per chunk type, per user
- **Custom eval scenarios** — we build AMB extensions that match your domain
- **Custom consolidation prompts** — tuned for your industry vocabulary
- **Migration assistance** — from LangChain / Mem0 / MemGPT / custom systems
- **Training** — engineering team workshops on memory-driven agent design

---

## Pricing Model

**Base fee:** starts at $3,000/mo. Includes 10 seats, 1M chunks stored, 500 eval runs/mo, 500 hosted consolidation jobs/mo.

**Scales with:**
- Seats (adds at $30/seat/mo after 10)
- Storage (adds at $0.20 per 1K chunks/mo after 1M)
- Hosted consolidation jobs (adds at $0.75/job after 500)
- Dedicated support tier (standard, premium, white-glove)
- Regional residency (single-region standard, multi-region +20%)

**Typical deals land in the $5K–$50K/mo range** depending on scale and compliance.

**Annual contracts** get a 10% discount. Multi-year (2+ years) get 15%.

---

## Security Posture

- SOC2 Type II (available under NDA)
- Annual 3rd-party penetration test (SANS-certified)
- Bug bounty program (private, scoped)
- Dependency scanning on every build (Snyk + Dependabot)
- Secrets scanning on every commit
- No training on customer data (contractual, not just policy)
- 100% reproducible builds

---

## Reference Architecture

```
┌──────────────────────────────────────────┐
│  Your VPC                                │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │  archon-memory-core OSS library  │    │
│  │  (in your application runtime)  │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│  ┌──────────────▼──────────────────┐    │
│  │  Control plane (our code,       │    │
│  │  your VPC, helm chart)          │    │
│  │  - Dashboard API                │    │
│  │  - Eval runner                  │    │
│  │  - Replay debugger              │    │
│  │  - Retention engine             │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│  ┌──────────────▼──────────────────┐    │
│  │  ChromaDB / your vector store   │    │
│  │  (never leaves your VPC)        │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│  ┌──────────────▼──────────────────┐    │
│  │  Ollama / vLLM cluster          │    │
│  │  (consolidation, your GPUs)     │    │
│  └─────────────────────────────────┘    │
└──────────────────────────────────────────┘
          │ (opt-in telemetry only)
          ▼
     Our observability cloud
     (anonymized metrics, no memory content)
```

**Data flow guarantee:** raw memory content never traverses our cloud. We see only anonymized metrics (chunk counts, eval scores, job counts). You turn even those off with a single config flag.

---

## What We Don't Do

- **We don't sell your data.** Period.
- **We don't train models on your memory.** Period.
- **We don't require exclusivity.** You can run our OSS alongside other memory systems. If you leave, your data stays where it is — we never held it.
- **We don't lock you in.** The OSS library is Apache 2.0 and self-sufficient. If Enterprise stops working for you, your production never stops — you just lose the observability layer.

---

## Procurement

- **Legal review:** we accept most vendor security questionnaires. Common answers pre-packaged.
- **Invoicing:** NET 30 standard, NET 60 on multi-year. Procurement portals (Ariba, Coupa) supported.
- **POC / pilot:** 30-day full-feature pilot with signed NDA. Converts to contract at end, or walks away.
- **Tax-exempt:** nonprofits and public-sector entities qualify with documentation.

---

## How to Buy

1. Email **enterprise@divergencerouter.com** with a brief description of your use case.
2. We schedule a 30-minute scoping call — no slideshow, no sales theater.
3. You get a quote + redlined DPA within 5 business days.
4. 30-day pilot → annual contract.

For everyone else, start with [Business or Team](PRICING.md). If you outgrow it, we're here.
