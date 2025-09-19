# AI4SE 2025  
Prof. Antonio Mastropaolo

Academic Year: 2025/2026

##Lab-01: Recommending Code Tokens via N-gram Models  

Your goal is to implement a probabilistic language model (i.e., **N-gram**) that can assist with code completion for software systems.  

<img width="400" height="75" alt="Screenshot 2025-09-17 at 11 05 02 AM" src="https://github.com/user-attachments/assets/b2698628-221d-41a7-a3de-1e99404ae789" />


Where \(t_n\) and \(t_{n-1}\) are sequences of code tokens.  

---


### Task Details
You are asked to implement an N-gram model to learn probabilities of token sequences based on their occurrences in your training data. This involves:  

- **Corpus Construction**: Use the dataset produced in Lab-00: [Java Mining Repositorie](https://github.com/ai4sewm-cloud/graduate-ai4se/blob/main/Labs/PLM/mining-java-methods.md) . The corpus structure has already been defined—please follow the format provided in Lab-00.

- **Context Window**: Treat the window size N as a tunable hyperparameter. Explore and compare results with different values (e.g., N = 3, 5, 7, 11, …).

- **Corpus Preprocessing**: Document how you cleaned and prepared your data. Show the steps you took to reduce noise (e.g., removing comments, imports, or formatting issues).  
  *Prof’s note:* Please make sure to comment your code.

- **Model Evaluation & Sampling**: Select an evaluation approach—either intrinsic (e.g., perplexity) or extrinsic (e.g., code completion success rate). You may also experiment with smoothing techniques.
Regardless of the choice, please read and implement the following:



Sampling from Your Model  

You are required to demonstrate your model’s behavior by sampling predictions on at least **1000 test inputs**.  

### What to Show
For each sampled input, provide:  
1. The **input context** (from your test set).  
2. The **predicted next tokens with probabilities**.  
3. At least one **sampled completion**.  

### Stopping Criteria
When sampling, stop generating tokens if:  
- The model predicts a special `<EOS>` token (if used),  
- The sequence is syntactically complete (e.g., ends with `}`), or  
- You reach a maximum token length (e.g., 50).  

---

### Example: Trigram Sampling  

**Input (from test set):**  
```java
for (int
```  

**Model predictions (given context “for ( int”):**  
- `"i"` → 0.62  
- `"j"` → 0.21  
- `"k"` → 0.09  
- `"value"` → 0.08  

→ Choose `"i"`  

**Next predictions (context “for ( int i”):**  
- `"="` → 0.73  
- `";"` → 0.18  
- others → 0.09  

→ Choose `"="`  

**Next predictions (context “for ( int i =”):**  
- `"0"` → 0.81  
- `"1"` → 0.12  
- others → 0.07  

→ Choose `"0"`  

**Sampled completion:**  
```java
for (int i = 0; i < n; i++) {
    // loop body
}
```  

### Deliverable: 
💡 Bonus Opportunity

You can earn up to 2.5 extra credits (added to your final grade) by submitting your lab.
This time, please push your code directly to our GitHub repository. I’ve created a dedicated folder for each of you under: ```Outputs/{Assignments|Labs}/your_github_name'''

Place all your work in Lab-00 and Lab-01. Students aiming for the bonus points will receive +1.5 credits for Lab-00 and +2.5 credits for Lab-01, provided the submission meets the requirements.

Your submission should include a fully working script that:
	1.	Takes as input a corpus in your chosen programming language (not limited to Java—you may generalize).
	2.	Builds and trains an N-gram model on this corpus.
	3.	Selects and outputs the best-performing N-gram model, based on your evaluation.
    3.1 For consistency feel free to use JsonL or CSV as you wish to store the predictions of the models


### Submission Instructions
	1.	Code & README: Submit your code along with a README that explains how to run it on a sample corpus.
	2.	Deadlines:
	    • Lab-00: due 09/17 at 11:59 PM
	    • Lab-01: due 09/21 at 11:59 PM
