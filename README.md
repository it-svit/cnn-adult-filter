
## Adult URL/page/document filtering by CNN model.

Project contain methods and classes that can help to create classification model 
(based on Keras+TensorFlow) for URLs/pages/documents that can be with different languages.

Also, project have pre-trained model for filtering URL/web-pages with adult content. 
This model receiving title, description, keywords and body text as inputs 
and return "Normal" or "Adult" based of received text. Model can classify 
**cs**, **de**, **en**, **es**, **fr**, **hu**, **it**, **nl**, **pl**, 
**pt**, **ru**, **tr**, **uk** languages.

Accuracy of a model - 95-99% (depending on language).

Prediction time - ~0.0045 sec per document.

Pre-trained model can be use via console and like a module in another project.

Console command:
```
python --title <string> --keywords <string> <string> ... <string> --description <string> --body-text <string>
```

where:

**--title** - title of a web-page;

**--description** - description of a web-page;

**--keywords** - keywords of a web-page;

**--body-text** - body text of a web-page.

For usages in other modules, please, look at `example.py`.

