# Business Report - Reducing Customer Churn with Targeted Retention

        ## Executive summary

        Churn affects 53 of 500 customers (10.6%). Every observed churn case is in the first 12 months, so the very strong model results should be treated as educational evidence rather than a production guarantee. The selected model identifies customers needing attention and converts each prediction into a risk band and suggested action. On the untouched test set it achieved recall 100.0%, precision 76.5%, F1 0.867, and ROC-AUC 1.000.

        ## Business problem

        Blanket discounts waste budget on customers unlikely to churn, while late action misses customers who need support. The proposed workflow prioritizes High-risk accounts for human outreach, gives Medium-risk accounts lighter proactive engagement, and avoids unnecessary discounts for Low-risk customers.

        ## What the data suggests

        Short tenure and contract/payment patterns are useful predictive signals. These patterns should guide conversation priorities, not be treated as proof of causality.

        ## Recommended operating policy

        1. **High risk:** contact within 48 hours, review service issues, and use an approved targeted offer.
        2. **Medium risk:** send proactive support and a contract-upgrade message; monitor response.
        3. **Low risk:** continue regular engagement without a retention discount.
        4. Review model recommendations with a human and record action/outcome for learning.
        5. Test offers with a randomized control group before scaling.

        ## Illustrative impact scenario

        | Scenario                |   Customers_Contacted |   Estimated_Churners_Reached |   Estimated_Net_Value_INR |
|:------------------------|----------------------:|-----------------------------:|--------------------------:|
| Blanket campaign        |                   500 |                           52 |                   -381600 |
| Model-targeted campaign |                    68 |                           52 |                    136800 |

        Assumptions: 35% of reached churners are saved, retained customer value is INR 12,000, and each offer costs INR 1,200. This compares strategies under explicit assumptions; it is not realized revenue.

        ## Implementation roadmap

        - **Pilot (2 weeks):** score a limited batch, validate contact workflow, and audit errors.
        - **Controlled test (4-6 weeks):** compare targeted treatment with a holdout group.
        - **Scale:** integrate the API, monitor model and campaign outcomes, and define owners.
        - **Review monthly:** measure precision, recall on matured labels, churn saved, cost per save, complaints, and subgroup differences.

        ## Decision requested

        Approve a small, human-reviewed pilot using the risk bands and capture treatment/outcome data. Do not commit full campaign budget until incremental lift is demonstrated.
