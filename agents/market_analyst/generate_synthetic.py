"""
Generates synthetic startup pitch JSONs across all 7 sectors for training data.
Run with --test for a quick 3-pitch check, or --full to generate the whole batch.
"""

import json
import sys
import time
import os
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "market_analyst"
OUTPUT_DIR = DATA_DIR / "synthetic_cases"


def _load_local_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if load_dotenv:
    load_dotenv()
else:
    _load_local_env_file()


api_key = os.getenv("OPENAI_API_KEY")
client = (
    OpenAI(
        api_key=api_key,
        base_url="https://llm-api.arc.vt.edu/api/v1/",
    )
    if api_key
    else None
)

# Each sector has a list of niches so the pitches don't all feel like the same startup.
SECTOR_NICHES = {
    "fintech": [
        "payroll automation for restaurant groups",
        "cross-border payments for freelancers in Southeast Asia",
        "spend management for construction subcontractors",
        "buy-now-pay-later for dental clinics",
        "expense reconciliation for remote-first teams",
        "invoice factoring for trucking fleets",
        "crypto tax filing for retail traders",
        "FX hedging tools for small importers",
        "embedded banking for gig economy platforms",
        "insurance compliance for commercial landlords",
        "real-time treasury for nonprofits",
        "lending underwriting for food trucks",
        "financial literacy app for Gen Z immigrants",
        "B2B card rewards for SaaS companies",
        "micro-investment platform for hourly workers",
        "payment links for independent therapists",
        "fraud detection for online marketplaces",
        "credit scoring for thin-file borrowers",
        "revenue-based financing for Shopify stores",
        "automated AP/AR for law firms",
        "digital wallets for domestic workers",
        "bank account switching assistant",
        "cashflow forecasting for corner stores",
        "student loan refinancing for nurses",
        "syndication tools for angel investors",
        "tax withholding automation for contractors",
        "insurance on-demand for event planners",
        "remittance rail for Ethiopia and Kenya",
        "point-of-sale lending for auto repair shops",
        "earned wage access for warehouse workers",
        "compliance monitoring for crypto exchanges",
        "financial advisor tools for recent immigrants",
        "pension plan automation for small businesses",
        "budgeting app for single parents",
        "subscription billing platform for boutique gyms",
        "vendor payment scheduling for retailers",
        "charitable giving integration for payroll",
        "insurance marketplace for independent contractors",
        "credit union modernization platform",
        "real estate escrow automation for agents",
        "corporate card for nonprofit organizations",
        "open banking API for community banks",
        "BNPL for veterinary care",
        "SMB accounting integrated with POS",
        "cross-border e-commerce payments",
        "digital banking for rural communities",
        "automated sales tax for e-commerce brands",
        "debt collection modernization for healthcare",
        "loyalty rewards tied to savings goals",
        "instant payout for marketplace sellers",
    ],
    "healthtech": [
        "remote monitoring for COPD patients",
        "mental health triage for college campuses",
        "care coordination for dialysis patients",
        "AI scribing for solo family practice doctors",
        "prior authorization automation for insurers",
        "home infusion therapy management",
        "nutrition tracking for bariatric surgery patients",
        "chronic pain management for aging adults",
        "dental imaging AI for rural clinics",
        "clinical trial matching for rare diseases",
        "pharmacy benefit management for self-insured employers",
        "pediatric therapy scheduling platform",
        "remote second opinion platform for oncology",
        "digital therapeutics for insomnia",
        "maternal health monitoring for high-risk pregnancies",
        "care navigation for Medicaid members",
        "musculoskeletal PT platform for employers",
        "hospital discharge planning software",
        "real-time sepsis detection for ICUs",
        "dermatology telediagnosis for rural communities",
        "nurse staffing marketplace for senior living",
        "electronic health record migrations for small clinics",
        "veteran mental health app",
        "caregiver support platform for dementia families",
        "hearing health platform for aging workers",
        "social determinants screening for Medicaid",
        "behavioral health intake automation",
        "clinical documentation for physical therapists",
        "postpartum depression screening app",
        "opioid recovery tracking platform",
        "oncology billing compliance tool",
        "specialty pharmacy coordination for rare diseases",
        "remote cardiac rehab program",
        "digital advance care planning tool",
        "connected inhaler for asthma management",
        "AI-powered radiology workflow tool",
        "employee mental health navigation app",
        "chronic disease management for uninsured patients",
        "care gap closure for Medicare Advantage plans",
        "clinical operations management for surgery centers",
        "mobile clinical trials for rural populations",
        "hospital supply chain optimization",
        "patient feedback platform for ambulatory care",
        "home sleep apnea testing",
        "revenue cycle AI for behavioral health",
        "remote dermatology monitoring for transplant patients",
        "ophthalmology AI for diabetic retinopathy",
        "medication adherence for transplant recipients",
        "occupational health platform for construction companies",
        "genetic counseling access platform",
    ],
    "ecommerce": [
        "resale marketplace for premium outdoor gear",
        "local artisan food marketplace",
        "rental platform for baby gear",
        "curated vintage fashion for Gen Z",
        "B2B wholesale marketplace for independent boutiques",
        "furniture rental for young professionals",
        "subscription box for indie bookstores",
        "marketplace for handmade ceramics and pottery",
        "peer-to-peer luxury watch trading",
        "returns management platform for DTC brands",
        "group buying for bulk groceries",
        "social commerce for Black-owned businesses",
        "sneaker authentication and trading platform",
        "e-commerce for international specialty foods",
        "art print marketplace connecting illustrators to buyers",
        "pre-loved electronics marketplace with warranty",
        "gifting platform for remote team celebrations",
        "custom apparel marketplace for micro-influencers",
        "farming equipment rental marketplace",
        "organic produce box with local farm connections",
        "marketplace for sustainable packaging materials",
        "second-hand kids clothing subscription",
        "rental platform for camera and photography gear",
        "live auction platform for estate sales",
        "marketplace for custom 3D printed products",
        "D2C herbal supplement brand with subscription",
        "luxury fragrance discovery subscription",
        "global food import marketplace for diaspora communities",
        "returns liquidation marketplace for brands",
        "marketplace for certified pre-owned sports equipment",
        "craft beer subscription club",
        "e-commerce for locally made furniture",
        "plant subscription and care subscription",
        "marketplace for indie game merchandise",
        "fitness equipment rental marketplace",
        "online marketplace for handmade jewelry",
        "dropshipping marketplace for sustainable goods",
        "ethnic beauty marketplace",
        "vintage homeware marketplace",
        "pet food subscription with vet-formulated recipes",
        "co-op buying platform for rural communities",
        "marketplace for upcycled home goods",
        "secondhand textbook exchange",
        "craft supply subscription for fiber artists",
        "audio equipment marketplace for musicians",
        "online marketplace for independent coffee roasters",
        "holiday gift curation platform",
        "custom shoe marketplace connecting cobblers to buyers",
        "marketplace for local honey and bee products",
        "DIY home repair parts marketplace",
    ],
    "saas": [
        "legal contract review for small law firms",
        "inventory forecasting for multi-location restaurants",
        "CRM built for independent insurance agents",
        "project management for architecture firms",
        "client portal for accounting firms",
        "fleet maintenance scheduling for landscapers",
        "compliance tracking for senior living facilities",
        "booking and scheduling for mobile pet groomers",
        "proposal and invoicing for freelance designers",
        "incident management for small IT teams",
        "permit tracking for construction companies",
        "candidate screening for hourly workforce hiring",
        "church management software",
        "studio management for tattoo shops",
        "employee training tracker for franchise operators",
        "content scheduling for newsletter writers",
        "donor management for community foundations",
        "quote-to-cash for HVAC companies",
        "customer feedback platform for product teams",
        "onboarding automation for remote-first startups",
        "approval workflow for marketing agencies",
        "field service management for solar installers",
        "academic advising platform for community colleges",
        "SaaS usage analytics for enterprise buyers",
        "event production management for AV companies",
        "AI-assisted grant writing for nonprofits",
        "document automation for real estate closings",
        "translation management for SaaS companies",
        "vendor risk management for mid-market companies",
        "knowledge base platform for customer success teams",
        "pricing intelligence for B2B sales teams",
        "board meeting management for credit unions",
        "student behavior tracking for K-12 schools",
        "revenue recognition automation for SaaS companies",
        "API documentation platform for developer tools",
        "product changelog and release notes platform",
        "QA testing management for agile teams",
        "manufacturing job shop scheduling tool",
        "remote equipment monitoring for construction sites",
        "shipping automation for Shopify merchants",
        "social media analytics for B2B marketers",
        "IT asset management for distributed teams",
        "feedback aggregation for product roadmaps",
        "localization platform for mobile apps",
        "subscription analytics for consumer apps",
        "brand asset management for marketing agencies",
        "HR compliance for multi-state employers",
        "contract lifecycle management for procurement teams",
        "business continuity planning for SMBs",
        "municipal permitting software for local governments",
    ],
    "media": [
        "audio drama subscription platform for commuters",
        "short-form documentary platform for independent filmmakers",
        "hyperlocal news app for neighborhood communities",
        "sports content platform for niche fan communities",
        "creator-owned newsletter infrastructure",
        "live trivia platform for sports bars",
        "interactive fiction app for Gen Z",
        "vertical video learning platform for tradespeople",
        "true crime podcast network for regional markets",
        "community radio platform for diaspora communities",
        "recipe video platform for home cooks",
        "behind-the-scenes fan access platform for musicians",
        "localized meme and culture platform",
        "language learning through pop culture content",
        "financial education content for first-generation investors",
        "children's audio story platform for bedtime",
        "short-form comedy platform for emerging writers",
        "fan community platform for indie artists",
        "niche sports stats and commentary app",
        "independent film crowdfunding and distribution",
        "wellness content platform for working mothers",
        "workplace culture and career advice media brand",
        "city-specific cultural events guide and app",
        "nostalgia content platform for millennials",
        "interactive personal finance education platform",
        "AI-personalized music discovery platform",
        "faith-based content streaming platform",
        "live shopping and entertainment hybrid platform",
        "community storytelling platform for elders",
        "political accountability journalism platform",
        "regional agriculture news platform",
        "esports content platform for college players",
        "travel storytelling platform for slow travelers",
        "daily briefing audio app for busy parents",
        "satirical news platform for younger voters",
        "craft and DIY tutorial streaming platform",
        "startup founder documentary series platform",
        "vintage film restoration and streaming service",
        "local restaurant discovery and food media brand",
        "personal finance reality content platform",
        "sustainability and climate storytelling platform",
        "rural lifestyle and homesteading content platform",
        "interactive sports betting analysis platform",
        "anime dubbing crowdfunding platform",
        "cross-cultural comedy content platform",
        "underground music archiving and discovery platform",
        "fan fiction publishing and monetization platform",
        "ambient soundscape subscription for productivity",
        "health and wellness audio meditation platform",
        "illustrated long-form journalism platform",
    ],
    "hardware": [
        "low-cost water quality sensor for rural farms",
        "wearable vibration sensor for industrial workers",
        "portable air quality monitor for urban cyclists",
        "smart beehive sensor for commercial apiaries",
        "modular home battery backup for apartments",
        "fleet tracking hardware for food delivery bikes",
        "RFID inventory system for independent pharmacies",
        "crop disease detection camera for smallholder farms",
        "remote patient vitals monitor for home hospice care",
        "low-cost ultrasound device for rural clinics",
        "structural health monitoring sensor for bridges",
        "smart irrigation controller for vineyards",
        "portable EV charger for apartment dwellers",
        "wearable fatigue monitor for long-haul truckers",
        "soil nutrient sensor for organic farms",
        "connected gym equipment for home users",
        "personal air purifier with real-time air quality data",
        "noise monitoring device for construction sites",
        "smart helmet with fall detection for cyclists",
        "indoor grow light controller for vertical farms",
        "pipeline leak detection sensor for utilities",
        "body camera with cloud sync for security guards",
        "modular hydroponic system for urban kitchens",
        "smart lock for shared storage spaces",
        "temperature and humidity logger for cold chain logistics",
        "portable ECG device for remote cardiac monitoring",
        "acoustic emission sensor for industrial machines",
        "livestock vital signs monitor",
        "smart thermostat for older apartment buildings",
        "wearable glucose monitor for non-diabetic fitness users",
        "autonomous seed-planting robot for small farms",
        "micro wind turbine for off-grid homes",
        "connected tool chest for construction job sites",
        "waste bin fill sensor for municipal waste management",
        "portable solar-powered router for disaster relief",
        "personal UV exposure tracker for outdoor workers",
        "3D food printer for commercial bakeries",
        "smart fishing rod sensor for anglers",
        "concrete curing monitor for construction projects",
        "robotic window cleaning device for mid-rise buildings",
        "vehicle diagnostics plug for independent mechanics",
        "wearable stress monitor for first responders",
        "hands-free barcode scanner for warehouse pickers",
        "smart mailbox sensor for package detection",
        "home fermentation monitoring device",
        "hydration sensor wristband for elderly care",
        "smart buoy for coastal water monitoring",
        "automated pet feeder with health tracking",
        "portable air compressor with IoT monitoring",
        "cold storage temperature alert system for restaurants",
    ],
    "food": [
        "ghost kitchen network for regional food brands",
        "meal prep delivery for bodybuilders",
        "snack subscription for office wellness programs",
        "fermented food brand targeting gut health consumers",
        "halal meal kit delivery service",
        "zero-waste restaurant supply platform",
        "farm-to-table catering marketplace for corporate events",
        "allergy-safe school lunch service",
        "single-origin coffee subscription from smallholder farms",
        "kombucha brewing kit subscription",
        "pre-portioned cooking kits for college dorms",
        "specialty sauce and condiment brand for foodies",
        "frozen African cuisine for diaspora households",
        "plant-based protein for Asian culinary traditions",
        "local brewery subscription box",
        "gourmet pet treat subscription",
        "community supported fishery platform",
        "curated spice subscription for home cooks",
        "smoothie delivery for health-conscious office workers",
        "Filipino street food brand for US markets",
        "functional mushroom supplement brand",
        "regional barbecue sauce delivery platform",
        "authentic Indian mithai delivery service",
        "school lunch catering with dietary tracking",
        "bulk dry goods delivery for urban households",
        "personalized sports nutrition meal delivery",
        "micro-batch hot sauce brand",
        "sourdough bread subscription from local bakeries",
        "corporate meal planning with dietary compliance",
        "specialty olive oil subscription from small growers",
        "seasonal cheese subscription from artisan creameries",
        "meal delivery for postpartum recovery",
        "protein snack bar brand for endurance athletes",
        "restaurant food waste reduction platform",
        "subscription for heirloom grain products",
        "food truck booking platform for corporate campuses",
        "catering management platform for wedding planners",
        "senior nutrition delivery service",
        "protein shake subscription for gym chains",
        "Filipino-Asian fusion meal kit",
        "meal prep delivery for people with diabetes",
        "online marketplace for local honey producers",
        "vegan comfort food brand",
        "restaurant software for nutrition labeling compliance",
        "custom birthday cake delivery marketplace",
        "alcohol-free cocktail subscription",
        "fresh pasta subscription from Italian-American makers",
        "meal delivery for shift workers and night staff",
        "personalized baby food subscription",
        "food tour booking platform for travelers",
    ],
}

SECTORS = list(SECTOR_NICHES.keys())

PITCH_COUNT_FULL = 50  # per sector, 350 total


def build_prompt(sector, niche, variation_hint):
    return f"""
You are writing a realistic early-stage startup pitch for training data.

Sector: {sector}
Niche: {niche}
Variation: {variation_hint}

Return a JSON object with exactly these keys:
{{
  "name": "startup name",
  "sector": "{sector}",
  "description": "2-3 sentence overview of what the startup does",
  "target_customer": "who pays for this and why",
  "pricing": "how they charge and rough price points",
  "traction": "early signs of product-market fit or lack of it",
  "team": "founder background, 1-2 sentences",
  "problem": "the specific problem being solved",
  "solution": "how the product solves it"
}}

Make the pitch feel like it was written by a real founder, not a marketing team.
Vary the quality based on the variation hint: {variation_hint}.
""".strip()


# I use three variation prompts so the dataset has a mix of strong, weak, and average pitches.
VARIATIONS = [
    "strong pitch with clear problem-solution fit and early traction",
    "weak pitch with vague value proposition and no real traction",
    "decent pitch but the market is crowded and team is thin",
    "promising product but the pricing model is unclear",
    "niche idea with a very specific customer but limited scale",
]


def generate_pitch(sector, niche, variation):
    if client is None:
        print(f"  no client configured, skipping {niche}")
        return None

    prompt = build_prompt(sector, niche, variation)

    try:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  failed for {niche}: {e}")
        return None


def run_sector(sector, niches):
    pitches = []

    for i, niche in enumerate(niches):
        variation = VARIATIONS[i % len(VARIATIONS)]
        print(f"  [{i + 1}/{len(niches)}] {niche}")

        pitch = generate_pitch(sector, niche, variation)
        if pitch:
            pitches.append(pitch)

        time.sleep(1)

    return pitches


def save_sector_file(sector, pitches):
    output_path = OUTPUT_DIR / f"{sector}_pitches.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pitches, f, indent=2)
    print(f"Saved {len(pitches)} pitches to {output_path.name}")


def run(test_mode=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pitches = []

    for sector in SECTORS:
        print(f"\nGenerating {sector} pitches...")

        niches = SECTOR_NICHES[sector]

        if test_mode:
            # just grab the first niche from the first 3 sectors so we can sanity check quickly
            if SECTORS.index(sector) >= 3:
                continue
            niches = niches[:1]

        else:
            niches = niches[:PITCH_COUNT_FULL]

        pitches = run_sector(sector, niches)
        all_pitches.extend(pitches)
        save_sector_file(sector, pitches)

    combined_path = OUTPUT_DIR / "all_synthetic_pitches.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_pitches, f, indent=2)

    print("Done generating synthetic pitches")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--full" in args:
        print("Running full generation (~350 pitches across 7 sectors)")
        run(test_mode=False)
    else:
        # defaults to test mode so you don't accidentally burn through the API
        print("Running test mode (3 pitches, quick check)")
        run(test_mode=True)
