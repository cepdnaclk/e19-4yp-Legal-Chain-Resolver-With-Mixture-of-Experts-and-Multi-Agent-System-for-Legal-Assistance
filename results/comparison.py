import pandas as pd
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [
    "What Are the Essential Elements of a Valid Contract?",
    "Do All Contracts Need to Be in Writing?",
    "What Constitutes a Breach of Contract?",
    "What Are the Remedies Available for Breach of Contract?",
    "When Does a Contract End?",
    "What Are Mutual Obligations in a Contract?",
    "Can a Contract Be Modified After It’s Been Signed?",
    "What Is the Difference Between a Condition Precedent and a Condition Subsequent in a Contract?",
    "What Happens if One Party Cannot Fulfill Their Contractual Obligations?"
]

original_answers = [
    "For a contract to be valid, it must have the following elements: - An agreement between parties - Intention to create legal obligations - A 'meeting of the minds' - Contractual capacity of the parties - Legality of purpose - Possibility of performance",
    "No, contracts can be either written or oral. However, some contracts must be in writing and signed by both parties to be legally binding. These include: - Contracts for the sale of land - Long leases (ten years or more) - Credit agreements - Contracts of suretyship - Contracts for executory donations - Marriage contracts",
    "A breach of contract occurs when one party fails to meet their obligations under the contract. This includes: - Non-performance - Partial performance - Repudiation",
    "The main remedies for a breach of contract are: - Specific performance - Interdict - Damages - Cancellation and damages",
    "A contract can end in several ways: - Fulfillment of obligations - Mutual agreement - Material breach - Death of a party - Expiry of the contract period",
    "Mutual obligations in a contract refer to the duties or responsibilities that both parties agree to perform. These obligations can include: - Performance of actions - Reciprocity",
    "Yes, a contract can be modified after it’s been signed, but this requires: - Mutual agreement - Consideration",
    "Condition precedent: A condition that must be met before the contract becomes effective. Condition subsequent: A condition that terminates the contract if it occurs after the contract has already been formed.",
    "If one party cannot fulfill their obligations, the contract may be breached, and the non-performing party may be held liable for damages or face cancellation of the contract depending on the terms and the nature of the failure to perform."
]

mas_answers = [
    "A valid contract requires essential elements, which include agreement between parties, intention to create legally binding obligations, contractual capacity, and lawful purpose. The requirements for a valid contract are outlined in legal frameworks.",
    "Contracts can be oral or written. However, specific contracts must be in writing and signed. Examples of contracts that require writing are long-term leases, contracts of suretyship, executory donations and marriage contracts. The Alienation of Land Act 68 of 1981 stipulates a writing requirement for land sales, and the National Credit Act 34 of 2005 stipulates a writing requirement for credit agreements.",
    "A breach of contract occurs when a party fails to fulfill their obligations. The Restatement (Second) of Contracts (1981) defines breach as failure to perform. Delivering defective goods constitutes partial performance and refusing to deliver goods signals repudiation.",
    "Remedies for breach aim to restore the aggrieved party’s position. Specific performance is court-ordered fulfillment of obligations. An interdict prevents further breaches. Damages compensate for losses. Cancellation with damages allows for contract termination.",
    "Contracts terminate through fulfillment of obligations, mutual agreement, material breach, death of a party, and expiry of contract period. Impossibility justifies termination.",
    "Mutual obligations refer to reciprocal duties. For example, in a contract, the seller delivers goods, and the buyer pays the price. Reciprocity is rooted in the Roman-Dutch law principle of synallagma.",
    "Contracts can be modified post-signing. The National Credit Act mandates written amendments. Mutual consent prevents unilateral alterations.",
    "A condition precedent is an event. A non-performing party bears consequences. Unilateral termination is permissible in specific circumstances.",
    "A non-performing party bears consequences. Impossibility justifies termination."
]

neo4j_answers = [
    "A valid contract requires capable parties, a clear offer and acceptance (mutual understanding), contractual capacity of the parties, a lawful purpose, and an intention to create legal relations. While a written form is listed, the query does not specify whether it is always required.",
    "No. The database indicates that not all contracts need to be in writing.",
    "A breach of contract can be constituted by repudiation, non-performance, or partial performance.",
    "The remedies available for breach of contract include damages, interdict, cancellation with damages, and specific performance.",
    "Contracts can end due to fulfillment of the contract's obligations, expiry of the contract's term, death of a party, fundamental non-performance, mutual agreement, or material breach.",
    "I'm sorry, but I couldn't find any information.",
    "Yes, a contract can be modified after it's been signed. This is called an amendment.",
    "I'm sorry, but the provided database information does not explain the difference.",
    "The provided information does not specify the answer."
]

grok_answers = [
    "A valid contract requires an agreement between parties, evidenced by an offer and acceptance, an intention to create legally binding obligations, a mutual understanding of terms, contractual capacity of parties, a lawful purpose, and terms capable of performance.",
    "Not all contracts need to be in writing; they can be oral or written, depending on the context, but specific contracts, such as land sales, credit agreements, long-term leases of ten years or more, contracts of suretyship, executory donations, and marriage contracts, must be in writing and signed to be enforceable under South African law.",
    "A breach of contract occurs when a party fails to fulfill their obligations, through non-performance, partial performance, or repudiation, such as delivering defective goods or refusing to deliver goods.",
    "Remedies for a breach of contract include specific performance, where the court orders fulfillment of obligations, an interdict to prevent further breaches, damages to compensate for losses, and cancellation with damages if the breach is material.",
    "A contract ends through fulfillment of obligations, mutual agreement, material breach, death of a party in personal service contracts, or expiry of the contract period.",
    "Mutual obligations are the reciprocal duties each party undertakes, often interdependent, such as the seller delivering goods and the buyer paying the price, ensuring neither party benefits without performing.",
    "A contract can be modified after signing if both parties agree and there is consideration, something of value exchanged to support the change, with written amendments required for credit agreements to ensure transparency.",
    "A condition precedent is an event that must occur before a contract becomes effective, such as obtaining loan approval before a property sale, while a condition subsequent terminates a contract if it occurs, like a lease ending if a property is rezoned.",
    "If a party cannot fulfill their contractual obligations, they may breach the contract, incurring liability for damages or facing cancellation, unless excused by the vis major doctrine for unforeseen events like natural disasters that make performance impossible."
]

deepseek_answers = [
    "A valid contract requires an agreement between parties, evidenced by offer and acceptance, and an intention to create legally binding obligations. There must be a mutual understanding or 'meeting of the minds' regarding the contract's terms. Parties must have contractual capacity, meaning they are of legal age and sound mind. The contract's purpose must be lawful, and its terms must be capable of performance. For example, a contract to sell illegal goods would be void.",
    "Not all contracts require written form; they can be oral or written, depending on the context. However, specific contracts under South African law must be in writing and signed to be enforceable. Examples include land sales under the Alienation of Land Act 68 of 1981, credit agreements under the National Credit Act 34 of 2005, long-term leases (ten years or more), contracts of suretyship, executory donations, and marriage contracts. This requirement ensures clarity and prevents disputes.",
    "A breach of contract occurs when a party fails to fulfill their obligations, whether through non-performance, partial performance, or repudiation (indicating an intent not to perform). Examples include delivering defective goods (partial performance) or refusing to deliver goods (repudiation). Such breaches disrupt the contractual balance and trigger remedies for the aggrieved party.",
    "Remedies for breach of contract include specific performance, where a court orders the fulfillment of obligations, particularly when monetary compensation is inadequate. An interdict may be used to prevent further breaches. Damages compensate for losses suffered, calculated based on the harm incurred. Cancellation with damages allows termination of the contract if the breach is material, as outlined in the Consumer Protection Act. These remedies ensure fairness and accountability in contractual agreements.",
    "A contract may end through fulfillment of obligations, mutual agreement, material breach, death of a party (in personal service contracts), or expiry of the contract period. The UNIDROIT Principles allow termination for fundamental non-performance, aligning with South African practices where material breaches justify cancellation. For instance, a lease agreement ends upon the specified term's expiry unless renewed.",
    "Mutual obligations in a contract refer to the reciprocal duties each party undertakes, which are often interdependent. For example, in a sale agreement, the seller must deliver the goods while the buyer must pay the agreed price. This reciprocity, rooted in the Roman-Dutch law principle of synallagma, ensures neither party benefits without performing their part of the agreement. These obligations maintain the contractual equilibrium between the parties.",
    "A contract can be modified after it is signed if both parties agree to the changes and there is consideration—something of value exchanged to support the modification. The National Credit Act requires written amendments for credit agreements to ensure transparency. Mutual consent prevents unilateral alterations, protecting the interests of both parties involved in the contract.",
    "A condition precedent is an event that must occur before a contract becomes effective, such as obtaining loan approval before a property sale can proceed. In contrast, a condition subsequent terminates a contract if a specified event occurs, such as a lease ending if the property is rezoned. These conditions provide flexibility in contract design and are recognized in legal frameworks like Kerr's The Principles of the Law of Contract.",
    "If a party cannot fulfill their contractual obligations, they may breach the contract, incurring liability for damages or facing cancellation. The Vis major doctrine, recognized in South African law, excuses performance if unforeseen events (e.g., natural disasters) make it impossible, as established in Peters v. Calamity (1980). Otherwise, the non-performing party bears the consequences, including potential remedies such as damages or specific performance sought by the aggrieved party."
]

gemini_answers = [
    "A valid contract requires several essential elements, including an agreement between parties, evidenced by an offer and acceptance, and an intention to create legally binding obligations. A mutual understanding, or 'meeting of the minds,' ensures both parties comprehend the contract's terms. Additionally, parties must have contractual capacity—being of legal age and sound mind—and the contract's purpose must be lawful, with terms capable of performance. These principles align with global standards, such as those in the UNIDROIT Principles of International Commercial Contracts (2016), which emphasize consensus and legality.",
    "Not all contracts require written form; they can be oral or written, depending on the context. However, specific contracts, particularly under South African law, must be in writing and signed to be enforceable, as stipulated by the Alienation of Land Act 68 of 1981 for land sales and the National Credit Act 34 of 2005 for credit agreements. Other examples include long-term leases (ten years or more), contracts of suretyship, executory donations, and marriage contracts. This requirement ensures clarity and prevents disputes, as highlighted in case law like Fourie v. Hansen (2000), which upheld the necessity of written agreements for land transactions.",
    "A breach of contract occurs when a party fails to fulfill their obligations, whether through non-performance, partial performance, or repudiation (indicating an intent not to perform). For example, delivering defective goods constitutes partial performance, while refusing to deliver goods signals repudiation. Such breaches disrupt the contractual balance and trigger remedies.",
    "Remedies for breach aim to restore the aggrieved party's position. Specific performance, a court-ordered fulfillment of obligations, is common when monetary compensation is inadequate. An interdict prevents further breaches, while damages compensate for losses, calculated based on the harm suffered. Cancellation with damages allows termination of the contract if the breach is material. These remedies ensure fairness and accountability.",
    "Contracts may terminate through several avenues: fulfillment of obligations, mutual agreement, material breach, death of a party (in personal service contracts), or expiry of the contract period. The UNIDROIT Principles (Article 7.3.1) allow termination for fundamental non-performance, aligning with South African practices where material breaches justify cancellation. For instance, a lease agreement ends upon the specified term’s expiry unless renewed.",
    "Mutual obligations refer to the reciprocal duties each party undertakes, often interdependent. For example, in a sale, the seller delivers goods while the buyer pays the price. This reciprocity, rooted in the Roman-Dutch law principle of synallagma, ensures neither party benefits without performing. These obligations maintain contractual equilibrium.",
    "Contracts can be modified post-signing, provided both parties agree and there is consideration—something of value exchanged to support the change. The National Credit Act mandates written amendments for credit agreements to ensure transparency. Mutual consent prevents unilateral alterations, safeguarding both parties' interests.",
    "A condition precedent is an event that must occur before a contract becomes effective, such as obtaining loan approval before a property sale. Conversely, a condition subsequent terminates a contract if it occurs, like a lease ending if a property is rezoned.",
    "If a party cannot fulfill their obligations, they may breach the contract, incurring liability for damages or facing cancellation. The Vis major doctrine, recognized in South African law, excuses performance if unforeseen events (e.g., natural disasters) make it impossible. Otherwise, the non-performing party bears the consequences."
]

rag_answers = [
    "The essential elements of a valid contract include an agreement between parties, evidenced by an offer and acceptance, and an intention to create legally binding obligations. There must be a mutual understanding, or “meeting of the minds,” ensuring both parties comprehend the contract’s terms. The parties involved must have the capacity to contract—legal age and sound mind—and the contract’s purpose must be lawful, with terms capable of performance.",
    "No, not all contracts require written form. They can be oral or written, depending on the context.",
    "A breach of contract happens when a party doesn't fulfill their obligations. This can be through not performing at all, performing only partially, or indicating they don't intend to perform.",
    "Remedies for breach aim to restore the aggrieved party’s position. Specific performance, a court-ordered fulfillment of obligations, is common when monetary compensation is inadequate. The termination of the contract is possible if the breach is material.",
    "Contracts may terminate through fulfillment of obligations, mutual agreement, material breach, death of a party (in personal service contracts), a condition subsequent that occurs, or expiry of the contract period.",
    "Mutual obligations refer to the reciprocal duties each party undertakes. For example, in a sale, the seller delivers goods while the buyer pays the price.",
    "Contracts can be modified post-signing, provided both parties agree and there is consideration.",
    "A condition precedent is an event that must occur before a contract becomes effective. A condition subsequent terminates a contract if it occurs.",
    "The non-performing party bears the consequences. Unilateral termination is permissible under specific circumstances, such as impossibility of performance. A breach of contract occurs when a party fails to fulfill their obligations."
]

# Encode original answers once
original_vecs = [model.encode([ans])[0] for ans in original_answers]

# Helper function using scipy cosine distance
def cosine_sim(vec1, vec2):
    return 1 - distance.cosine(vec1, vec2)

answer_dict = {
    'MAS': mas_answers,
    'Neo4j': neo4j_answers,
    'Grok': grok_answers,
    'DeepSeek': deepseek_answers,
    'Gemini': gemini_answers,
    'RAG': rag_answers
}

results = []
for i, orig_vec in enumerate(original_vecs):
    row = {'Question': questions[i]}
    for method, answers in answer_dict.items():
        method_vec = model.encode([answers[i]])[0]
        row[method] = cosine_sim(orig_vec, method_vec)
    results.append(row)

df = pd.DataFrame(results)
df.set_index('Question', inplace=True)

print("Similarity Scores for each question:")
print(df)

# Calculate average similarity score for each method
avg_scores = df.mean(axis=0).to_frame(name='Average Similarity Score')

print("\nAverage Similarity Scores by Method:")
print(avg_scores)