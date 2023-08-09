# This is ok because it's tiny, < 100 lines

FEW_SHOT = """
Q: Why do I eat just enough when eating breakfast?
Answer Choices:
(A) gas
(B) full stomach
(C) feeling satisfied (CORRECT)
(D) have energy
(E) bloated
A: (C) (CORRECT)
Reasoning: The answer must be a reason for eating just enough when eating breakfast. Eating just enough is a way to feel satisfied. Therefore, the answer is feeling satisfied (C).

Q: How is a dog likely to communicate with another dog?
Answer Choices:
(A) reproduce
(B) bark (CORRECT)
(C) do many things
(D) clone
(E) jump up
Answer: (B) (CORRECT)
Reasoning: The answer must be a way for a dog to communicate with another dog. Dogs bark to communicate with each other. Therefore, the answer is bark (B).

Q: If a person is trying to weasel out of punishment for a crime, where are they likely to be?
Answer Choices:
(A) court room (CORRECT)
(B) cherry tree
(C) chicken coop
(D) natural history museum
(E) jail
A: (A) (CORRECT)
Reasoning: The answer must be a place where a person would be trying to get out of punishment for a crime. The answer is court room (A).

Q: What would easily hide a snake?
Answer Choices:
(A) living room of a house
(B) kew gardens
(C) terrarium
(D) thick forest (CORRECT)
(E) tropical forest
A: (D) (CORRECT)
Reasoning: The answer must be something that would hide a snake. The answer is thick forest (D).

Q: The person didn’t like the varying size of the cake slices handed out, she demand more what?
Answer Choices:
(A) fairness (CORRECT)
(B) candies
(C) compliments
(D) well fed
(E) arguements 2
A: (A) (CORRECT)
Reasoning: The answer must be a reason for the person to demand more cake. The person is not happy with the size of the cake slices. Therefore, the answer is fairness (A).

Q: In America, the courthouse of any given town has a likely location, where is it?
Answer Choices:
(A) electrical circuit
(B) getting married
(C) capital city
(D) center of town (CORRECT)
(E) michigan
A: (D) (CORRECT)
Reasoning: The answer must be a location where a courthouse is located. The capital city is the location of the capital of a state. Therefore, the answer is center of town (D).

Q: Where might I find a bowl nestled among other bowls in the kitchen?
Answer Choices:
(A) refrigerator
(B) kitchen top
(C) cupboard (CORRECT)
(D) dishwasher
(E) apartment
Answer: (C) (CORRECT)
Reasoning: The answer must be a place where a bowl is found. The kitchen is a place where bowls are found. Therefore, the answer is cupboard (C).

Q: Which state has blue ridge mountains that are called Blue Mountain?
Answer Choices:
(A) appalachians
(B) virginia
(C) pennsylvania (CORRECT)
(D) north carolina
(E) montana 
A: (C) (CORRECT)
Reasoning: The answer must be a state with mountains. The answer is Pennsylvania (C).

Q: Where could you find many radio studio?
Answer Choices:
(A) radio station
(B) country
(C) clear area
(D) office building
(E) large city (CORRECT)
A: (E) (CORRECT)
Reasoning: The answer must be a place where many radio studios are located. Radio studios are used to broadcast radio programs. Therefore, the answer is large city (E).

Q: Where would someone bring you a cup? 
Answer Choices: 
(A) apartment
(B) closet
(C) restaurant (CORRECT)
(D) table
(E) party
A: (C) (CORRECT)
Reasoning: The answer must be a place where someone would bring you a cup. A restaurant is a place where people bring cups. Therefore, the answer is restaurant (C).
"""


def main():
    import transformers
    t = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    print(len(t(FEW_SHOT).input_ids))
    print(t)

if __name__ == "__main__":
    main()