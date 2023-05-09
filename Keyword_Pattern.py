import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QLineEdit, QProgressBar
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QBrush, QColor
from PyQt5.QtCore import QSize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import cProfile

# Download the required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create a WordNetLemmatizer object
lemmatizer = WordNetLemmatizer()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle("Keyword Extractor")

        # Create the main layout
        layout = QHBoxLayout()

        # Create the left layout for the keywords and n-gram size input
        left_layout = QVBoxLayout()
        
        # Create a frame for the left layout and set its frame shape and shadow
        left_frame = QtWidgets.QFrame()
        left_frame.setFrameShape(QtWidgets.QFrame.Box)
        left_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        left_frame.setLayout(left_layout)
        left_frame.setFixedWidth(300)
        
        
        layout.addWidget(left_frame)
        left_layout.addSpacing(120)

        # Create the label for the keyword patterns
        self.pattern_label = QLabel("Keyword Patterns:")
        
         # Increase font size of label
        font = self.pattern_label.font()
        font.setPointSize(font.pointSize() * 1.5)
        self.pattern_label.setFont(font)
       
        
        left_layout.addWidget(self.pattern_label)

        # Create a scrolling text box for displaying the keyword patterns
        self.keyword_textbox = QTextEdit()
        self.keyword_textbox.setReadOnly(True)
        size_hint = self.keyword_textbox.sizeHint()
        new_size_hint = 250
        self.keyword_textbox.setFixedWidth(new_size_hint)
        
        left_layout.addWidget(self.keyword_textbox)

        # Create the right layout for the text box and extract button
        right_layout = QVBoxLayout()
        # Create a frame for the left layout and set its frame shape and shadow
        right_frame = QtWidgets.QFrame()
        right_frame.setFrameShape(QtWidgets.QFrame.Box)
        right_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        right_frame.setLayout(right_layout)
        
        layout.addWidget(right_frame)

        # Create the button for extracting keywords
        self.extract_button = QPushButton("Find Patterns")
        self.extract_button.clicked.connect(self.extract_keywords)
        
        # Set button color to jade green
        # Round corners of button
        self.extract_button.setStyleSheet("""
            QPushButton {
                background-color: #00A86B;
                color: white;
                border-radius: 10px;
                font-size: 20px;
            }
            QPushButton:pressed {
                background-color: #008B57;
            }
            """)
        size_hint = self.extract_button.sizeHint()
        new_size_hint = QSize(int(size_hint.width() * 1.5), int(size_hint.height() * 2))
        self.extract_button.setFixedSize(new_size_hint)
        
        right_layout.addWidget(self.extract_button)

        # Create a progress bar to indicate the progress of extract_keywords
        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)

        # Create the text box for user input
        right_layout.addSpacing(25)
        self.textbox = QTextEdit()
        self.textbox.setStyleSheet("border-top: 1px solid black;")
        right_layout.addWidget(self.textbox)


        # Create an empty QLabel object
        spacer = QLabel()

        # Set the fixed height of the QLabel object to create spacing
        spacer.setFixedHeight(400)

        # Add the QLabel object to the layout above left frame
        layout.insertWidget(layout.indexOf(left_frame), spacer)
        # Set the main layout
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def extract_keywords(self):
        # Get the text from the text box
        text = self.textbox.toPlainText()

        # Set the n-gram size and top N value
        min_n=2
        max_n=3

        # Define a function to generate n-grams from text
        def ngrams(text, n):
            tokens = text.split()
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
            return ngrams
            
        # Define a function to preprocess the text
        def preprocess(text):
            # Lowercase the text
            text = text.lower()
            
            # Tokenize the text
            tokens = word_tokenize(text)
            
            # Remove punctuation and special characters
            tokens = [token for token in tokens if token.isalpha()]
            
            # Remove stop words
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
            # Lemmatize the tokens
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # Return the preprocessed text
            return ' '.join(tokens)

        # Preprocess the text and create a TfidfVectorizer object
        text_to_tfidf = preprocess(text)
        vectorizer = TfidfVectorizer(ngram_range=(min_n,max_n))

        # Fit and transform the text to get a TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform([text_to_tfidf])

        # Get the feature names (words) from the vectorizer
        words = vectorizer.get_feature_names_out()

        # Extract the top N keywords for each paragraph
        for i, paragraph in enumerate([text_to_tfidf]):
            # Get the row corresponding to the current paragraph
            row = tfidf_matrix[i]

            # Create a PorterStemmer object
            stemmer = PorterStemmer()

            # Group similar words together and keep only one representative from each group
            word_groups = {}
            for index in row.indices:
                word = words[index]
                stemmed_word = ' '.join([stemmer.stem(token) for token in word.split()])
                if stemmed_word not in word_groups:
                    word_groups[stemmed_word] = (word,row[0,index])
                else:
                    if row[0,index] > word_groups[stemmed_word][1]:
                        word_groups[stemmed_word] = (word,row[0,index])

            # Get top N words from each group based on their TF-IDF values
            all_words = [word for word,value in sorted(word_groups.values(), key=lambda x: -x[1])[:10]]

            # Count the occurrences of each keyword in the text
            #word_counts = Counter(ngrams(text_to_tfidf.lower(), max_n))
            all_word_counts = []
            for word in all_words:
                stemmed_word = ' '.join([stemmer.stem(token) for token in word.split()])
                count = 0
                for n in range(min_n, max_n + 1):
                    for ngram in ngrams(text_to_tfidf.lower(), n):
                        stemmed_ngram = ' '.join([stemmer.stem(token) for token in ngram.split()])
                        if stemmed_word == stemmed_ngram:
                            count += 1

                all_word_counts.append(count)

            # Stem each keyword and count its occurrences
            top_word_counts = []
            prog_count = 0
            for word in all_words:
                stemmed_word = ' '.join([stemmer.stem(token) for token in word.split()])
                count = 0
                for ngram in ngrams(text_to_tfidf.lower(), max_n):
                    stemmed_ngram = ' '.join([stemmer.stem(token) for token in ngram.split()])
                    if stemmed_word == stemmed_ngram:
                        count += 1
                
                top_word_counts.append(count)
                # Update the progress bar value after each keyword is processed
                self.progress_bar.setValue(int((prog_count+1)/len(all_words)*100))
                prog_count+=1
            
            # Filter words that have more than 2 occurrences
            filtered_words = [word for word, count in zip(all_words, all_word_counts) if count >= 2]
                
            # Update the keyword text box with the top N keywords and their counts
            print(filtered_words)
            print(all_word_counts)
            keyword_text = '\n'.join([f"{word} ({count})" for word, count in zip(filtered_words, all_word_counts)])
            self.keyword_textbox.setPlainText(keyword_text)
                    
            QMessageBox.information(self, "Keywords Extracted", "The keywords have been extracted and are displayed on the left side of the window.")
            
    def highlight_keyword(self, label):
        # Get the keyword from the label text
        keyword = label.text().split(":")[0]

        # Create a QTextCharFormat object to specify the highlight format
        fmt = QTextCharFormat()
        fmt.setBackground(QBrush(QColor("yellow")))

        # Create a QTextCursor object to highlight occurrences of the keyword
        cursor = self.textbox.textCursor()
        cursor.setPosition(0)
        cursor.movePosition(QTextCursor.Start)

        # Check if the keyword is already highlighted
        highlighted = False
        while True:
            # Find the next occurrence of the keyword
            pos = self.textbox.toPlainText().find(keyword, cursor.position())
            if pos < 0:
                break

            # Select the keyword
            cursor.setPosition(pos)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(keyword))

            # Check if the keyword is highlighted
            if cursor.charFormat().background().color().name() == QColor("yellow").name():
                highlighted = True
                break

        # Reset the cursor position
        cursor.setPosition(0)
        cursor.movePosition(QTextCursor.Start)

        while True:
            # Find the next occurrence of the keyword
            pos = self.textbox.toPlainText().find(keyword, cursor.position())
            if pos < 0:
                break

            # Select and highlight or unhighlight the keyword
            cursor.setPosition(pos)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(keyword))
            if highlighted:
                cursor.setCharFormat(QTextCharFormat())
            else:
                cursor.mergeCharFormat(fmt)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()