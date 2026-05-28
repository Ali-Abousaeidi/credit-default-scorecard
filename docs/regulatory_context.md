# Regulatory Context

This project is a portfolio model, not a production credit decisioning system
or legal assessment. The purpose is to show awareness of the governance context
around credit scoring and probability-of-default models.

## EU AI Act

The European Commission's AI Act Service Desk lists AI systems used to evaluate
the creditworthiness of natural persons or establish their credit score under
Annex III high-risk systems, except systems used for financial fraud detection:

https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3

The EBA's 2025 note on AI Act implications for EU banking and payments also
states that AI used to evaluate creditworthiness or establish the credit score
of natural persons is high-risk, and that the AI Act adds safeguards for those
systems:

https://eba.europa.eu/sites/default/files/2025-11/d8b999ce-a1d9-4964-9606-971bbc2aaf89/AI%20Act%20implications%20for%20the%20EU%20banking%20sector.pdf

Project implication: the README and model documentation should cover data
governance, model purpose, validation, stability monitoring, explainability,
human review, and limitations.

## Banking Model Governance

The ECB's 2025 guide-to-internal-models FAQ explains that the guide sets out
how the ECB understands rules for internal models used by directly supervised
institutions to compute own funds requirements for credit, market, and
counterparty credit risk. It also notes expectations around model risk
management, data governance, machine learning techniques, explainability, model
use, internal validation, and internal audit:

https://www.bankingsupervision.europa.eu/press/other-publications/publications/html/ssm.faq_guide_internal_models_2025~baf433d505.en.html

Project implication: even though this is not an IRB regulatory-capital model,
the documentation should mimic bank practice: objective, scope, data, method,
assumptions, validation, monitoring, and limitations.
