require 'brains'


f = File.open('corpus.txt','r')

@words = []

f.each_line do |line|
  @words += line.downcase.split(" ")
end
f.close

@word_dict = @words.uniq.sort

def to_1hot(value, width)
  arr = (0...width).collect do
    0.0
  end

  arr[value] = 1.0
  arr
end

def word_to_binary(w)
  w = @word_dict.first unless @word_dict.index(w.downcase)
  to_1hot(@word_dict.index(w.downcase), 15)
end

def onehot_to_word(arr)
  index = 0
  highest_val = 0.0
  arr.each_with_index do |v, i|
    if v > highest_val
      index = i
      highest_val = v
    end
  end

  @word_dict[index]
end

p @words.size
p @word_dict.size
p word_to_binary(@words.first)

nn = Brains::Net.create(15, 15, 2, { neurons_per_layer: 15,
      learning_rate: 0.5,
      recurrent: true,
      output_function: :sigmoid,
      error: :cross_entropy
     })

# train on corpus


input = @words.collect { |w|  word_to_binary(w) }
expected_output = @words.rotate.collect { |w|  word_to_binary(w) }

training_data = [[input, expected_output]]

p training_data

result = nn.optimize_recurrent(training_data, 0.01, 100_000_000, 10) { |i, error|
  puts "#{i} #{error}"
}

p @words
p @word_dict

input  = @words[1]

sentence = (1..10).collect do
  puts input
  r = nn.feed(word_to_binary(input))
  puts r
  output = onehot_to_word(r)
  input = output
  output
end.join(' ')

puts sentence
