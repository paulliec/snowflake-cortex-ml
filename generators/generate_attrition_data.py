"""
Generate synthetic employee attrition data for hard-to-staff roles.
Produces realistic patterns: churn correlates with low manager ratings,
stale raises, high overtime, and poor performance trajectory.
"""

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# -- Role configs: department, salary range, target churn rate, overtime range
ROLES = {
    "Pilot": {
        "department": "Aviation Operations",
        "salary": (95_000, 210_000),
        "churn_rate": 0.18,
        "overtime": (5, 35),
        "remote_eligible": "N",
    },
    "Cardiologist": {
        "department": "Cardiology",
        "salary": (280_000, 520_000),
        "churn_rate": 0.16,
        "overtime": (10, 40),
        "remote_eligible": "N",
    },
    "ICU Nurse": {
        "department": "Critical Care",
        "salary": (68_000, 115_000),
        "churn_rate": 0.13,
        "overtime": (15, 50),
        "remote_eligible": "N",
    },
    "Data Engineer": {
        "department": "Engineering",
        "salary": (105_000, 185_000),
        "churn_rate": 0.09,
        "overtime": (0, 20),
        "remote_eligible": "Y",
    },
    "Security Engineer": {
        "department": "Engineering",
        "salary": (115_000, 195_000),
        "churn_rate": 0.08,
        "overtime": (0, 25),
        "remote_eligible": "Y",
    },
    "Anesthesiologist": {
        "department": "Anesthesiology",
        "salary": (300_000, 480_000),
        "churn_rate": 0.16,
        "overtime": (10, 35),
        "remote_eligible": "N",
    },
    "Flight Paramedic": {
        "department": "Aviation Operations",
        "salary": (55_000, 95_000),
        "churn_rate": 0.13,
        "overtime": (10, 45),
        "remote_eligible": "N",
    },
}

REGIONS = ["Midwest", "Southeast", "Northeast", "West", "Central"]

# -- Exit survey templates grouped by reason
EXIT_REASONS = {
    "compensation": [
        "Honestly, the pay just wasn't competitive anymore. I got an offer for {pct}% more and couldn't justify staying.",
        "I liked the work but the salary hadn't moved in {years} years. At some point you have to look out for yourself.",
        "Compensation was below market. I raised it with my manager twice and nothing changed.",
        "The benefits were fine but base pay was {amt}k under what I was seeing elsewhere. Had to make a move.",
        "I didn't want to leave but financially it didn't make sense to stay. The raise cycle felt performative.",
        "Pay was stagnant. I watched newer hires come in at higher comp than me after {years} years here.",
    ],
    "work_life_balance": [
        "The hours were just unsustainable. I was doing 60+ hour weeks for months straight.",
        "I missed too many family events. My kid's recital, anniversaries. The schedule was relentless.",
        "Burnout. Pure burnout. I needed to step away before it affected my health more than it already had.",
        "On-call rotations were brutal. I'd get paged at 3am and still be expected in at 7.",
        "There was no real boundary between work and personal time. Texts at midnight, weekend expectations.",
        "I asked for schedule flexibility after my second child and was told it wasn't possible. Found somewhere it was.",
    ],
    "better_opportunity": [
        "Got recruited for a role that was a clear step up. More scope, better title, stronger team.",
        "An opportunity came along that I couldn't pass up. Better aligned with where I want my career to go.",
        "I wasn't actively looking but a recruiter reached out with something too good to ignore.",
        "The new role gives me exposure to things I'd never get to touch here. It was a growth decision.",
        "Took a position at a company doing more interesting work in my specialty area.",
    ],
    "management": [
        "My manager and I just didn't see eye to eye. Feedback was inconsistent and I never knew where I stood.",
        "Leadership turnover killed morale. Three managers in two years, each with different priorities.",
        "I didn't feel supported. Brought up concerns multiple times and they went nowhere.",
        "The micromanagement was suffocating. Every decision needed three approvals.",
        "My manager took credit for my work on the {project} initiative. That was the last straw.",
        "No mentorship, no development conversations. Just task assignment and status updates.",
    ],
    "burnout": [
        "I was running on empty. The pace never let up and there was always another fire to fight.",
        "Compassion fatigue is real in this line of work. I needed a reset.",
        "The emotional toll caught up with me. I started dreading coming in and that's not who I am.",
        "Staffing shortages meant I was doing the work of two people for over a year. Something had to give.",
        "I loved the mission but the workload was destroying my mental health.",
    ],
    "relocation": [
        "My spouse got transferred to {city}. We tried to make it work remotely but the role required on-site.",
        "Moved back home to be closer to aging parents. Wasn't the company's fault, just life.",
        "Relocated for my partner's career. Would have stayed otherwise.",
        "Family situation changed and I needed to be in a different part of the country.",
    ],
}

CITIES = ["Denver", "Austin", "Chicago", "Seattle", "Boston", "Nashville", "Phoenix", "Atlanta"]
PROJECTS = ["Q3 migration", "capacity planning", "the new scheduling system", "patient intake redesign"]


def generate_exit_survey(role: str) -> str:
    """Generate a realistic exit survey response. Weights vary by role."""
    # weight reasons by role type
    if role in ("Pilot", "Flight Paramedic"):
        weights = {"compensation": 0.25, "work_life_balance": 0.30, "better_opportunity": 0.15,
                    "management": 0.10, "burnout": 0.15, "relocation": 0.05}
    elif role in ("Cardiologist", "Anesthesiologist"):
        weights = {"compensation": 0.30, "work_life_balance": 0.20, "better_opportunity": 0.20,
                    "management": 0.10, "burnout": 0.15, "relocation": 0.05}
    elif role == "ICU Nurse":
        weights = {"compensation": 0.20, "work_life_balance": 0.15, "better_opportunity": 0.10,
                    "management": 0.15, "burnout": 0.30, "relocation": 0.10}
    else:
        weights = {"compensation": 0.25, "work_life_balance": 0.15, "better_opportunity": 0.25,
                    "management": 0.15, "burnout": 0.10, "relocation": 0.10}

    reasons = list(weights.keys())
    probs = list(weights.values())

    # pick 1-2 reasons, combine for longer responses sometimes
    n_reasons = random.choices([1, 2], weights=[0.6, 0.4])[0]
    chosen = np.random.choice(reasons, size=n_reasons, replace=False, p=probs)

    parts = []
    for reason in chosen:
        template = random.choice(EXIT_REASONS[reason])
        text = template.format(
            pct=random.randint(15, 40),
            years=random.randint(2, 5),
            amt=random.randint(10, 45),
            city=random.choice(CITIES),
            project=random.choice(PROJECTS),
        )
        parts.append(text)

    survey = " ".join(parts)

    # occasionally add a softener
    if random.random() < 0.3:
        softeners = [
            " I wish things had worked out differently.",
            " No hard feelings — it was a tough call.",
            " I'd consider coming back if things changed.",
            " Great people here, just not the right fit anymore.",
        ]
        survey += random.choice(softeners)

    return survey


def generate_records(n: int) -> pd.DataFrame:
    """Generate n employee records with realistic attrition patterns."""
    records = []

    # distribute roles roughly: more nurses and engineers, fewer physicians
    role_weights = {
        "Pilot": 0.12,
        "Cardiologist": 0.08,
        "ICU Nurse": 0.20,
        "Data Engineer": 0.18,
        "Security Engineer": 0.14,
        "Anesthesiologist": 0.08,
        "Flight Paramedic": 0.10,
    }
    # pad remaining to generic split
    total_w = sum(role_weights.values())
    role_names = list(role_weights.keys())
    role_probs = [role_weights[r] / total_w for r in role_names]

    roles_assigned = np.random.choice(role_names, size=n, p=role_probs)

    for i, role in enumerate(roles_assigned):
        cfg = ROLES[role]
        emp_id = f"EMP-{i + 1:05d}"
        name = fake.name()
        region = random.choice(REGIONS)

        # tenure: gamma distribution skewed toward 2-5 years
        tenure = round(np.clip(np.random.gamma(2.5, 2.0), 0.5, 15.0), 1)

        salary_low, salary_high = cfg["salary"]
        # salary correlates with tenure
        tenure_pct = min(tenure / 15.0, 1.0)
        base = salary_low + (salary_high - salary_low) * (tenure_pct * 0.6 + random.uniform(0, 0.4))
        salary = int(round(base, -3))

        # performance: right-skewed (most people are 3-4)
        perf = int(np.clip(np.random.normal(3.6, 0.9), 1, 5))
        manager_rating = int(np.clip(np.random.normal(3.4, 1.0), 1, 5))

        promotions = random.choices([0, 1, 2, 3], weights=[0.45, 0.35, 0.15, 0.05])[0]

        flight_hours = None
        if role == "Pilot":
            flight_hours = int(np.clip(np.random.normal(650, 200), 100, 1200))

        ot_low, ot_high = cfg["overtime"]
        overtime = round(np.clip(np.random.normal((ot_low + ot_high) / 2, (ot_high - ot_low) / 4), 0, 80), 1)

        days_since_raise = int(np.clip(np.random.exponential(300), 30, 1500))

        team_size = random.randint(4, 25)
        remote = cfg["remote_eligible"]

        hire_date = datetime.now() - timedelta(days=int(tenure * 365))
        hire_date_str = hire_date.strftime("%Y-%m-%d")

        # -- Churn decision: base rate adjusted by risk factors
        # tuned so overall rate lands ~18%, pilots ~25%, physicians ~22%
        churn_prob = cfg["churn_rate"]

        # low manager rating increases churn
        if manager_rating <= 2:
            churn_prob += 0.08
        elif manager_rating <= 3:
            churn_prob += 0.02

        # stale raises increase churn
        if days_since_raise > 600:
            churn_prob += 0.06
        elif days_since_raise > 400:
            churn_prob += 0.02

        # high overtime increases churn
        ot_mid = (ot_low + ot_high) / 2
        if overtime > ot_mid * 1.4:
            churn_prob += 0.04

        # low performance + low promotions = disengaged
        if perf <= 2 and promotions == 0:
            churn_prob += 0.06

        # short tenure slight bump
        if tenure < 1.5:
            churn_prob += 0.03

        # cap it
        churn_prob = min(churn_prob, 0.55)

        churned = random.random() < churn_prob

        term_date = None
        exit_text = None
        if churned:
            # termination sometime in last 2 years
            days_employed = int(tenure * 365)
            term_offset = random.randint(1, min(days_employed, 730))
            term_date = (hire_date + timedelta(days=days_employed - term_offset)).strftime("%Y-%m-%d")
            exit_text = generate_exit_survey(role)

        records.append({
            "employee_id": emp_id,
            "employee_name": name,
            "role": role,
            "department": cfg["department"],
            "region": region,
            "tenure_years": tenure,
            "salary": salary,
            "performance_score": perf,
            "manager_rating": manager_rating,
            "promotions_last_3_years": promotions,
            "flight_hours_ytd": flight_hours,
            "overtime_hours_monthly": overtime,
            "days_since_last_raise": days_since_raise,
            "team_size": team_size,
            "remote_eligible": remote,
            "hire_date": hire_date_str,
            "termination_date": term_date,
            "churned": "Y" if churned else "N",
            "exit_survey_text": exit_text,
            "exit_survey_sentiment": None,  # populated by Cortex AI_SENTIMENT later
        })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    """Print churn summary stats."""
    total = len(df)
    churned = df[df["churned"] == "Y"]
    active = df[df["churned"] == "N"]

    print(f"\n{'='*60}")
    print(f"Generated {total} employee records")
    print(f"Overall churn rate: {len(churned)/total:.1%}")
    print(f"{'='*60}")

    print(f"\nChurn rate by role:")
    print(f"{'Role':<25} {'Total':>6} {'Churned':>8} {'Rate':>8}")
    print(f"{'-'*50}")
    for role in sorted(ROLES.keys()):
        role_df = df[df["role"] == role]
        role_churned = role_df[role_df["churned"] == "Y"]
        rate = len(role_churned) / len(role_df) if len(role_df) > 0 else 0
        print(f"{role:<25} {len(role_df):>6} {len(role_churned):>8} {rate:>8.1%}")

    print(f"\nAverage tenure (years):")
    print(f"  Active:  {active['tenure_years'].mean():.1f}")
    print(f"  Churned: {churned['tenure_years'].mean():.1f}")

    print(f"\nAverage manager rating:")
    print(f"  Active:  {active['manager_rating'].mean():.2f}")
    print(f"  Churned: {churned['manager_rating'].mean():.2f}")

    print(f"\nAverage days since last raise:")
    print(f"  Active:  {active['days_since_last_raise'].mean():.0f}")
    print(f"  Churned: {churned['days_since_last_raise'].mean():.0f}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic employee attrition data")
    parser.add_argument("--records", type=int, default=2000, help="Number of records to generate")
    parser.add_argument("--output-file", type=str, default="data/synthetic_attrition_data.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"Generating {args.records} synthetic employee records...")
    df = generate_records(args.records)
    df.to_csv(args.output_file, index=False)
    print(f"Saved to {args.output_file}")

    print_summary(df)


if __name__ == "__main__":
    main()
