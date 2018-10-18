import re
import unicodedata
# import emoji

# def demojify(string):
# 	string = string.lower()
# 	emoji_pattern = re.compile("["
# 	                       u"\U0001F600-\U0001F64F"  # emoticons
# 	                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
# 	                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
# 	                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
# 	                       u"\U00002702-\U000027B0"
# 	                       u"\U000024C2-\U0001F251"
# 	                       "]+", flags=re.UNICODE)

# 	string = emoji_pattern.sub(r'', string).split()
# 	return (string)

# def demojify_v2(text):
#     allchars = [str for str in text]
#     emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
#     clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
#     return clean_text

def demojify_v3(text):
    # For each word in sentence we encode it to ascii and decode it and check if its equal to its original form.
    output = [i for i in text.split() if (i == i.encode("ascii", errors="ignore").decode())]
    return ' '.join(output)


def demojify_v4(text):
	output = []

	for i in text.split():
		# Check if word is english or not
		if (i == i.encode("ascii", errors="ignore").decode()):
			output.append(i)
		else:
			# Splitting by emoji from text
			emo = re.split(r'([^\w\s,])', i)
			# Remove empty strings
			emo = list(filter(None, emo))
			# Eliminating duplicates
			seen = set()
			emo = [x for x in emo if not (x in seen or seen.add(x))]
			# For every unicode get name 
			for e in emo:
				try:
					output = output + (unicodedata.name(e).lower().split())
				except:
					output.append(e)

	return ' '.join(output)
