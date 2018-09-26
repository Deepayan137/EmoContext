import re
import emoji

def demojify(string):
	string = string.lower()
	emoji_pattern = re.compile("["
	                       u"\U0001F600-\U0001F64F"  # emoticons
	                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
	                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	                       u"\U00002702-\U000027B0"
	                       u"\U000024C2-\U0001F251"
	                       "]+", flags=re.UNICODE)

	string = emoji_pattern.sub(r'', string).split()
	return (string)

def demojify_v2(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text