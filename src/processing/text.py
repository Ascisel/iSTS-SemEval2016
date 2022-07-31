def clear_sentence(text):
    text = text.replace('//', '')
    text = text.strip()
    return text


def clear_alignment(text):
    new_text = []

    for chunk in text.split('\n'):
        chunks = chunk.split('//')

        alignment = chunks[0]
        class_1 = chunks[1].strip()
        class_2 = chunks[2].strip()

        chunks_text = chunks[3].split('---')
        chunks_text[0] = chunks_text[0].strip()
        chunks_text[1] = chunks_text[1].strip()

        alignments = alignment.split('---')
        alignment_source = alignments[0].strip()
        alignment_translation = alignments[1].strip()

        alignment_source = tuple(map(int, alignment_source.split(' ')))
        alignment_translation = tuple(map(int, alignment_translation.split(' ')))

        row = [class_1, class_2, alignment_source, alignment_translation, chunks_text]
        new_text.append(row)

    return new_text


def source_to_words(text):
    text.replace('\n', '')
    chunks = text.split(':')
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        chunk = chunk[1:]
        chunk = chunk.strip()
        result.append(chunk)
    return result
