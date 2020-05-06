from http.server import BaseHTTPRequestHandler, HTTPServer
from json import dumps
import pickle
import numpy
import scipy.special

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

    def activation_function(self, x):
        return scipy.special.expit(x)

    def inverse_activation_function (self, x):
        return scipy.special.logit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

class RequestHandler(BaseHTTPRequestHandler):

  def _send_cors_headers(self):
      """ Sets headers required for CORS """
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
      self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")

  def send_dict_response(self, d):
      """ Sends a dictionary (JSON) back to the client """
      self.wfile.write(bytes(dumps(d), "utf8"))

  def do_OPTIONS(self):
      self.send_response(200)
      self._send_cors_headers()
      self.end_headers()

  def do_GET(self):
      self.send_response(200)
      self._send_cors_headers()
      self.end_headers()
      response = {}
      response["status"] = "OK"

      values = list()
      for i in range(10):
          t = numpy.zeros(10) + 0.01
          t[i] = 0.99

          # get image data
          image_data = n.backquery(t) * 255
          image_data = image_data.astype(int)
          print(','.join(map(str, image_data.tolist())).replace('[', '').replace(']', ''))
          values.append(' '.join(map(str, image_data.tolist())).replace('[', '').replace(']', ''))
          response["result" + str(i)] = values[i]
          # matplotlib.pyplot.imsave("rrs.png", image_data.reshape((28, 28)), cmap='Greys')



      self.send_dict_response(response)

  def do_POST(self):
      self.send_response(200)
      self._send_cors_headers()
      self.send_header("Content-Type", "application/json")
      self.end_headers()

      dataLength = int(self.headers["Content-Length"])
      data = self.rfile.read(dataLength)

      data_array = data.decode("utf-8")

      all_values = data_array.split(',')



      # correct answer is first value
      # scale and shift the inputs
      if len(all_values) == 784:
          inputs = (numpy.asfarray(all_values[0:]) / 255.0 * 0.99) + 0.01
          image = numpy.asfarray(all_values[0:]).reshape((28, 28))
          # query the network
          outputs = n.query(inputs)
          # the index of the highest value corresponds to the label
          label = numpy.argmax(outputs)

      if len(all_values) == 785:
          inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
          image = numpy.asfarray(all_values[1:]).reshape((28, 28))

          # query the network
          outputs = n.query(inputs)
          # the index of the highest value corresponds to the label
          label = numpy.argmax(outputs)
          print(all_values[0])
          if label != all_values[0]:
              targets = numpy.zeros(10) + 0.01
              targets[int(all_values[0])] = 0.99
              n.train(inputs, targets)
              pickle.dump(n, open('neuron_network_values_1000.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

          print("train")




      response = {}
      response["status"] = "OK"
      response["result"] = str(label)
      response["percent"] = str(int(outputs[label]*100))
      self.send_dict_response(response)


n = pickle.load(open('neuron_network_values_1000.pkl', 'rb'))




print("Запуск сервера")
httpd = HTTPServer(("127.0.0.1", 9000), RequestHandler)
print("Сервер работает на порту 9000")
httpd.serve_forever()