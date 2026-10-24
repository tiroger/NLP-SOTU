"""Policy-topic taxonomy for paragraph classification.

The build classifies paragraphs with the supervised model in
sotu/validate_topics.py, which uses the descriptions below as training
anchors and as the rubric Claude labels with. Without Claude labels it falls
back to this module's method: the topic whose centroid (the mean embedding of
its descriptions) is most similar, if the similarity clears MIN_SIMILARITY,
else "Other". Descriptions mix modern and 19th-century phrasing so both eras
land in the right place.
"""

import numpy as np

MIN_SIMILARITY = 0.30

TOPICS = {
    "Economy & jobs": [
        "Unemployment, jobs, wages and economic growth for working families.",
        "Inflation, recession, prices and the cost of living.",
        "Manufacturing, industry, business and the prosperity of the country.",
        "The depression in business and the distress of labor.",
    ],
    "Taxes & budget": [
        "Federal spending, the budget deficit and the national debt.",
        "Tax cuts, tax reform and the burden of taxation.",
        "The revenues of the Treasury, receipts and expenditures for the fiscal year.",
        "Reducing the public debt and appropriations by Congress.",
    ],
    "Trade & tariffs": [
        "Tariffs, imports, exports and trade agreements with other countries.",
        "Duties on imports and the protection of domestic industry.",
        "Commerce with foreign nations and reciprocity treaties.",
    ],
    "Banking & currency": [
        "The currency, coinage of gold and silver, and the monetary standard.",
        "The national bank, paper money, banking and credit.",
        "The Federal Reserve, interest rates and financial regulation of Wall Street.",
    ],
    "Foreign affairs": [
        "Diplomatic relations, treaties and negotiations with foreign governments.",
        "Our alliances, the United Nations and leadership in the world.",
        "Our ministers abroad, claims against foreign powers and neutrality.",
    ],
    "Defense & war": [
        "The army, the navy and the strength of our armed forces.",
        "War, military operations, troops and our servicemembers.",
        "Terrorism, national security, nuclear weapons and defense spending.",
        "Veterans, soldiers and sailors who served in the war.",
    ],
    "Health care": [
        "Health insurance, affordable health care, Medicare and Medicaid.",
        "Prescription drug prices, hospitals, doctors and patients.",
        "Public health, disease, epidemics, vaccines and medical research.",
    ],
    "Education": [
        "Schools, teachers, students and education reform.",
        "College tuition, student loans and job training.",
        "Universities, learning and the education of our children.",
    ],
    "Immigration": [
        "Immigration, the border, asylum and illegal immigrants.",
        "Naturalization of aliens and the admission of immigrants.",
        "Securing the southern border and enforcing immigration law.",
    ],
    "Energy & environment": [
        "Climate change, clean energy, pollution and the environment.",
        "Oil, gas, energy independence and electricity.",
        "Conservation of forests, water, public lands and natural resources.",
    ],
    "Crime & justice": [
        "Crime, police, drugs, prisons and public safety.",
        "The courts, judges, the Department of Justice and law enforcement.",
        "Gun violence, criminal justice reform and the rule of law.",
    ],
    "Civil rights": [
        "Civil rights, equality, discrimination and voting rights.",
        "Slavery, emancipation and the freedmen.",
        "Equal rights for women, minorities and all citizens.",
    ],
    "Social welfare": [
        "Social Security, retirement, pensions and the elderly.",
        "Poverty, welfare, housing and help for families in need.",
        "Child care, paid leave and support for working parents.",
    ],
    "Agriculture": [
        "Farmers, agriculture, crops and farm prices.",
        "The Department of Agriculture and rural communities.",
    ],
    "Infrastructure": [
        "Roads, bridges, highways, railroads and transportation.",
        "Internal improvements, canals, harbors and rivers.",
        "Broadband, airports, ports and public works.",
    ],
    "Native American affairs": [
        "The Indian tribes, treaties with the Indians and Indian lands.",
        "Removal of the Indians and the frontier settlements.",
        "Tribal nations and Native American communities.",
    ],
    "Science & technology": [
        "Scientific research, space exploration and innovation.",
        "Technology, the internet, artificial intelligence and computers.",
    ],
    "Government & administration": [
        "The civil service, executive departments and government reorganization.",
        "The postal service, the Post-Office Department and public offices.",
        "Reform of government, waste, efficiency and federal employees.",
        "The Constitution, Congress, elections and the powers of government.",
    ],
    "Territories & expansion": [
        "The acquisition and annexation of new territory such as Louisiana, Florida, Texas or Alaska.",
        "Admission of new States into the Union and governments for the Territories.",
        "Puerto Rico, the Philippines, Hawaii, Guam and our island possessions.",
    ],
}


def centroids(embed) -> tuple[list[str], np.ndarray]:
    names = list(TOPICS)
    mats = [embed(TOPICS[n]).mean(axis=0) for n in names]
    c = np.stack(mats)
    return names, c / np.linalg.norm(c, axis=1, keepdims=True)


def assign(paragraph_embeddings: np.ndarray, names: list[str], cents: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return each paragraph's topic label and its similarity to that topic."""
    sims = paragraph_embeddings @ cents.T
    best = sims.argmax(axis=1)
    score = sims[np.arange(len(best)), best]
    return [names[b] if score[i] >= MIN_SIMILARITY else "Other" for i, b in enumerate(best)], score
