# Pricing

**The OSS library is free forever and always will be.** Paid tiers exist for teams shipping agents to production who need observability, team features, and hosted services on top of the OSS core.

Paid tiers launch **Q3 2026** after the longitudinal benchmark + OSS adoption milestones. The tiers below are the committed structure. Join the waitlist at [divergencerouter.com](https://divergencerouter.com) to be invited to the beta.

---

## The Ladder

| Tier | Price | Target |
|---|---|---|
| **Free OSS** | $0 | Individual devs, researchers, self-hosted indie shops |
| **Solo Dev** | **$19/mo** | Indie devs running agents in production, side projects |
| **Team** | **$99/mo** | 3–10 seat startup engineering teams |
| **Business** | **$499/mo** | 10+ seat engineering orgs with compliance needs |
| **Enterprise** | **from $3,000/mo** | Regulated industries, high-volume, custom deployment |

**No per-agent pricing.** Workspace base fee + usage overages. Power users don't get punished for scaling agents.

---

## What's In Each Tier

| Feature | Free | Solo $19 | Team $99 | Business $499 | Enterprise $3K+ |
|---|---|---|---|---|---|
| Full OSS library | ✓ | ✓ | ✓ | ✓ | ✓ |
| Local ChromaDB + Ollama consolidation | ✓ | ✓ | ✓ | ✓ | ✓ |
| AMB eval harness (run locally) | ✓ | ✓ | ✓ | ✓ | ✓ |
| LangChain / LlamaIndex adapters | ✓ | ✓ | ✓ | ✓ | ✓ |
| Memory health dashboard | — | ✓ | ✓ | ✓ | ✓ |
| Email alerts (health score drops) | — | ✓ | ✓ | ✓ | ✓ |
| Hosted eval runs | — | 5/mo | 50/mo | 500/mo | Unlimited |
| Replay debugger | — | — | ✓ | ✓ | ✓ |
| Shared workspaces | — | — | ✓ (3 seats) | ✓ (10 seats) | Unlimited |
| Webhook alerts | — | — | ✓ | ✓ | ✓ |
| Hosted async consolidation | — | — | — | 50 jobs/mo | Unlimited |
| Retention policies per namespace | — | — | — | ✓ | ✓ |
| SOC2 Type I | — | — | — | ✓ | ✓ |
| SLA | Community | Community | Community | 99.5% | 99.95% |
| VPC / private deploy | — | — | — | — | ✓ |
| SSO / SAML | — | — | — | — | ✓ |
| Signed DPA / BAA | — | — | — | — | ✓ |
| Audit logs | — | — | — | 90 days | Configurable |
| Premium support | — | Community | Community | Business-hours | 24/7 dedicated |

---

## Usage Overages

All paid tiers include generous starting quotas. If you exceed them, overage rates apply.

| Overage | Solo | Team | Business |
|---|---|---|---|
| Additional 1K chunks stored | $0.50 | $0.50 | $0.40 |
| Additional eval run | $2.00 | $1.00 | $0.75 |
| Additional hosted consolidation job | n/a | n/a | $1.00 |

Overages are billed monthly via Stripe. Your dashboard shows live usage against quota. Set hard caps if you want to block overages.

---

## Why these tiers, and not others

We iterated on pricing with ~15 hours of super-council deliberation and validated against comparable developer tools (Linear, Sentry, PostHog, Retool). Key decisions:

- **$19 Solo vs. $39:** Lowering the floor gets indie devs and side-project builders onto usage. They become evangelists and feed the benchmark flywheel.
- **$99 Team vs. $199:** A 5x gap between Solo and Team was a canyon. 5x between Team and Business works because the buyer profile changes (individual vs. procurement).
- **$499 Business tier (new):** SOC2 + SLA + retention is the compliance tier most engineering orgs need. Before they talk to enterprise sales, they need this to exist.
- **$3K Enterprise floor (published):** We refuse to list "Custom" with no anchor. If $3K is too much, you're not ready. If you need more, we'll price by load and compliance requirements.

---

## What stays free, forever

Not paywall bait. The OSS library is the product, and it's complete:

- All 13 memory chunk types + salience priors
- Full 6-stage retrieval pipeline
- Cross-encoder re-ranking
- MMR diversity
- Entity graph with 2-hop expansion
- Working memory buffer
- Nightly consolidation (requires your Ollama)
- Full AMB eval harness
- LangChain + LlamaIndex adapters
- All future retrieval algorithm improvements

If your production agent runs happily on the OSS library and your own cron, we want it there. The paid tier is for teams who need to see what's happening, prove it to auditors, and move faster together.

---

## Unit economics commitment

We publish cost-per-customer estimates in the [unit economics doc](docs/UNIT_ECONOMICS.md) after each pricing review. If the math stops working, we raise prices — we don't hide inflation in reduced quotas.

**Current targets:**
- 80%+ gross margin on Solo + Team
- 70%+ on Business
- 60%+ on Enterprise (reflects support cost)

---

## Questions

- **Can I self-host the paid features?** Business and Enterprise can run the observability/eval backend inside their own VPC. Solo and Team use our cloud.
- **Is my data used to train models?** Never. ToS prohibits it, and hybrid mode means raw memory doesn't leave your environment at all on any tier.
- **GDPR / CCPA?** Full deletion within 30 days of request. DPA on Enterprise. We're not in the business of hoarding customer data — the value is in the software, not the data.
- **What happens if I stop paying?** All paid features gracefully disable. OSS library keeps working forever. No lock-in.

---

## Enterprise details

See [ENTERPRISE.md](ENTERPRISE.md) for VPC deployment, SSO/SAML, DPA/BAA, audit logs, and custom retention. Contact enterprise@divergencerouter.com for a quote.
