
arg_list = argv();
train = arg_list{1};
dev = arg_list{2};

train = readdata(train, '%f%f%f');
dev = readdata(dev, '%f%f');

[rows,cols] = size(train);
nin = cols - 1;
nhidden = nin+1;
nout = 1;
weight_interval_value = 0.05;

%initialize weights to uniform random samples between -0.05 to 0.05
Wih = randmatrix(nin+1, nhidden, -weight_interval_value, weight_interval_value);
Who = randmatrix(nhidden+1, nout, -weight_interval_value, weight_interval_value);

prediction = NN(train, dev, 1, 10000, nin, nhidden, nout, Wih, Who);

rows = size(prediction)(2);
for i=1:rows
    if prediction(i) < 0.1
        printf("0\n");
    else
        printf("1\n");
    end
end
