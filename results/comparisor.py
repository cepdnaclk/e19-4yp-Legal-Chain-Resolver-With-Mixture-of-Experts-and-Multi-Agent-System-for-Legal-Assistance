import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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

def preprocess(text):
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

original_processed = [preprocess(ans) for ans in original_answers]
mas_processed = [preprocess(ans) for ans in mas_answers]
grok_processed = [preprocess(ans) for ans in grok_answers]
deepseek_processed = [preprocess(ans) for ans in deepseek_answers]
gemini_processed = [preprocess(ans) for ans in gemini_answers]

# Compute cosine similarity using TF-IDF
vectorizer = TfidfVectorizer()
cosine_sim_results = []

for i in range(len(questions)):
    # Combine original and model answers for vectorization
    texts = [original_processed[i], mas_processed[i], grok_processed[i], deepseek_processed[i], gemini_processed[i]]
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Compute similarity between original (index 0) and others
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    cosine_sim_results.append({
        "Question": questions[i],
        "Original Answer": original_answers[i],
        "MAS": cosine_sim[0],
        "GROK": cosine_sim[1],
        "DeepSeek": cosine_sim[2],
        "Gemini": cosine_sim[3]
    })
df = pd.DataFrame(cosine_sim_results)
print(df)