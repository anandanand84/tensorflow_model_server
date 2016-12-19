# tensorflow_model_server

Serve tensorflow models using tensorflow serving tensorflow model server.

## How to serve

Export your model using exporter in tensorflow session bundle 

```
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

def export_model(sess, export_path, export_version, x, y , probability):
    export_saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(export_saver)
    model_exporter.init(sess.graph.as_graph_def(), named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'classes': y , 'probability' : probability})})
    model_exporter.export(export_path, tf.constant(export_version), sess)
    print("Export Finished!")
    
#export_model(sess, '/model/', 1 , x , output, probability)
```

Run your model using tensorflow_model server by mapping /model directory to the container and specifing the model base path

```
docker run anandanand84/tensorflow_model_server --model_base_path=/model
```

For complete example look at the example.py in the repository.

You can also build a new container from this base by adding your model files in the container.
