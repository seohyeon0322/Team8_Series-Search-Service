from search_service import get_corpus, get_scores, processing_score

print("===========Series Serch Service===========")
while(1):
    user_input = input("Enter a content you want to find. (enter q to finish): ")
    if (user_input == 'q'):
        break

    corpus = get_corpus('Harry Potter')
    text = corpus.copy()
    sim_scores = get_scores(corpus, user_input)

    answer = processing_score(user_input, sim_scores, relation=True)
    if (answer == 8):
        print("Please enter better input.")
        print("1. This content may not in Harry Potter.")
        print("2. his content may cover the entire series.")
    else: 
        print("It appears in series: "+ str(answer))
        print()
        answer = input("Do you want to keep searching? (y/n) ")
        if (answer == 'y'):
            pass
        else:
            break
    print()
