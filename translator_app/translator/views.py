import tensorflow as tf
from django.shortcuts import render
import numpy as np

# Load the model from the SavedModel format
model = tf.keras.models.load_model('translator/translation_model (3).keras')


# Create your views here
def translate(request):
    responses = ""
    if request.method == 'POST':
        text = request.POST['Name']
        responses = predicting(text)

    return render(request, "index.html", {'responses': responses})

def predicting(data):
    max_decoded_sentence_length = 20

    def translate(input_sentence):

        string_tensor_1 = tf.convert_to_tensor(input_sentence, dtype=tf.string)

        # Step 3: Expand dimensions to create a batch of size 1 (optional)
        string_tensor = tf.expand_dims(string_tensor_1, axis=0)

        decoded_sentence = "startofseq"

        string_tensor_2 = tf.convert_to_tensor(decoded_sentence, dtype=tf.string)

        # Step 3: Expand dimensions to create a batch of size 1 (optional)
        decoded_sentence = tf.expand_dims(string_tensor_2, axis=0)

        for i in range(max_decoded_sentence_length):

            # y_proba = model([string_tensor, decoded_sentence])
            y_proba = model.predict((string_tensor, decoded_sentence))[0, i,:]

            predicted_word_id = tf.argmax(y_proba).numpy()

            loaded_vect_sp = model.layers[3]

            sampled_token = loaded_vect_sp.get_vocabulary()[predicted_word_id]

            decoded_sentence += " " + sampled_token
            if sampled_token == "endofseq":
                break
        return decoded_sentence

    seed_text = data
    generated = translate(seed_text)
    cleaned_tensor = tf.strings.regex_replace(generated, "startofseq|endofseq", "")
    generated_final = np.array(cleaned_tensor)
    plain_text = generated_final[0].decode('utf-8')
    final_result = [f"The English sentence is:  {data}",f"The corresponding Spanish translation would be :  {plain_text}"]
    return final_result
