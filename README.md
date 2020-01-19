# huddl_assignment

## Steps
#### 1. Normalize the data: 
Read the data and extract the Subject and Message of the email. The rest of the fields are not necessary.
Once the subject and message are extracted, parse them to remove punctuations and invalid characters such as "new line (\n)".

#### 2. Generate heuristic rules for determining if a sentence is actionable or not:
Priority is given to subject and then to the message. The reason being a subject is a one-line summary of the entire email.
Hence, a subject containing actionable sentence is more relevant.
##### Rules 
    1. Named entity resolution
    2. Part-of-speech tags extraction
        We consider only these POS tags 
        NN (for nouns)
        NNP (pronouns)
        PRP (personal pronouns) 
        VB (present tense verbs)
        VBD (past tense verb)
        VBZ (3rdperson singular verb)
        VBN (past participle)
        VBP (non-3rd person singular)
    3. Co-reference resolution
    4.  

POS Rules considered.

1. (MD) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
2. (PRP/PRP$) (MD) (VB/VBD/VBG/VBN/VBP/VBZ)
3. (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
4. (VB/VBD/VBG/VBN/VBP/VBZ) (PRP/PRP$) (NN/NNS)
5. (VB/VBD/VBG/VBN/VBP/VBZ) (CC) (PRP/PRP$) (NN/NNS)
6. (VB/VBD/VBG/VBN/VBP/VBZ) (CC) (PRP/PRP$) (IN) (NN/NNS)