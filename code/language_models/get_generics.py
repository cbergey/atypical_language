from google.cloud import aiplatform
import pandas as pd

project_id = "hs-social-interaction-lab"  
location = "europe-west4"        
endpoint_id = "mg-endpoint-8c366c82-94b9-4c2d-bdd5-749bdf4e1cd4" 

aiplatform.init(project=project_id, location=location)
endpoint = aiplatform.Endpoint(endpoint_id)

df = pd.read_csv("../../data/all_corpora_adj_noun_conc_utts.csv")
#generics prompt
#prepend_prompt = "You are doing a task in which you judge whether a sentence is a generic: whether makes a claim about a category as a whole without quantifying how many members of a category have a certain feature. For instance, 'Birds lay eggs' and 'Dog is man's best friend' and 'The ostrich is a tall species of bird' are generics. On the other hand, 'This soup is hot' and 'Some paper has gridlines' and 'Most shells have ridges' are not generics--they refer to a specific entity or quantify ('some','most') the number of members of a category that have a feature. Remember, you're looking for sentences that express a generic quality about a category; the sentence doesn't need to correctly identify a feature, just attempt to express a generic feature. Answer simply with 'generic' or 'non-generic'. Here is the sentence:\n"
#quantified majority feature prompt
prepend_prompt = "You are doing a task in which you judge whether a sentence makes a claim about a majority feature: whether makes a claim about a category and some feature or aspect the majority of its members have. For instance, 'All birds lay eggs' and 'Most dogs are friendly' and 'In general, ostriches tend to be tall' express something about a majority feature. On the other hand, 'This soup is hot' and 'Some paper has gridlines' and 'Few shells have spiral shapes' do not--they refer to a specific entity or they claim that a non-majority of the category have a feature. Remember, you're looking for sentences that express a majority quality about a category; the sentence doesn't need to correctly identify a majority feature, just attempt to express a majority feature. Answer simply with 'majority' or 'non-majority'. Here is the sentence:\n"

#responses = ("generic","non-generic")
responses = ("majority","non-majority")

def get_rating(utterance, depth = 0):
	print(utterance)
	response = endpoint.predict(
		instances = [{"prompt": prepend_prompt + " " + utterance + "\nNow answer with 'majority' or 'non-majority'. Answer:\n"}]
		)
	response = response.predictions[0]
	print("full response: " + response)
	
	response = response.partition('\n')[0].lower().strip()
	response = response.partition('.')[0].lower().strip()
	response = response.partition('!')[0].lower().strip()
	response = response.partition(',')[0].lower().strip()
	print("truncated response: " + response)
	if response not in responses:
		print("Failed with response: " + response)
		if depth < 5:
			return get_rating(utterance, depth + 1)
		else:
			return "uncategorizable"
	else:
		return response

df['majority_llama_70b'] = df.apply(lambda x: get_rating(x['sentence']).strip(), axis = 1)
df.to_csv('../../data/majority_llama_judgments.csv', index = None)