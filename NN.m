function prediction = NN(train, dev, learning_factor, weight_interval_value)

    rows = size(train)(1);
    cols = size(train)(2);

    train = train(2:rows,:);
    rows = rows - 1;

    nin = cols - 1;
    nhidden = nin+1;
    nout = 2;
    ntot = nin + nhidden + nout;
    
    %initialize weights to uniform random samples between -0.05 to 0.05
    Wih = randMatrix(nin+1, nhidden, -weight_interval_value, weight_interval_value);
    Who = randMatrix(nhidden+1, nout, -weight_interval_value, weight_interval_value);


    function M = randMatrix(rows, cols, lower, upper)
        M = lower - (upper-lower)*rand(rows, cols);
    end

    function [Oh Oo] = update_output(X, Oh, Oo)
        %calculating the output for the hidden nodes
        %input -> O(1,nin)
        for i = 1:nhidden
            Oh(i) = Wih(1,i) + dot(X,Wih(2:nin+1,i));
        end

        %calculating the output values for the output nodes
        %input -> O(nin+1, nhidden)
        for i = 1:nout
            Oo(i) = Who(1,i) + dot(Oh,Who(2:nhidden+1,i));
        end
        
        ret = 0;
    end

    function [Eh Eo] = backpropagate_error(y, Oh, Oo)
        %propagate error for output units
        for i=1:nout
            o = sigmoid(Oo(i));
            Eo(i) = o * (1-o) * (y-o);
        end

        %propagate error for hidden units
        for i=1:nhidden
            o = sigmoid(Oh(i));
            Eh(i) = o * (1-o) * dot(Who(i,:), Eo);
        end
        ret = 0;
    end

    function ret = update_weights(X, Oh, Oo, Eh, Eo)
        for i=1:nin
            for j=1:nhidden
                Wih(i,j) = Wih(i,j) + learning_factor*Eh(j)*X(i);
            end
        end
        for i=1:nhidden
            for j=1:nout
                Who(i,j) = Who(i,j) + learning_factor*Eo(j)*Oh(i);
            end
        end
        ret = 0;
    end

    function ret = NN_train()

        Oh = zeros(1,nhidden);
        Oo = zeros(1,nout);

        Eh = zeros(1,nhidden);
        Eo = zeros(1,nout);

        %Backpropagation with one hidden layer 
        %Stochastic gradient descent to update weights
        for iteration = 1:500
            sigma = 0;
            for row = 1:rows
                X = [train(row,1:cols-1)];
                y = train(row,cols);
                
                [Oh Oo] = update_output(X, Oh, Oo);
                [Eh Eo] = backpropagate_error(y, Oh, Oo, Eh, Eo);
                update_weights(X, Oh, Oo, Eh, Eo);

                for i=1:nout
                    sigma = sigma + 0.5*(y-Oo(i))^2;
                end
            end
            if mod(iteration, 10) == 0
                printf('%f\n', sigma);
            end
        end
    end
    
    function O = NN_test()
        
        rows = size(dev)(1);
        cols = size(dev)(2);
        ret = zeros(1,rows-1);
        for i = 1:rows-1
            Oh = zeros(1,nhidden);
            Oo = zeros(1,nout);
            X = dev(i+1,1:cols);

            [Oh, Oo] = update_output(X, Oh, Oo);
            ret(i) = mean(Oo);
        end
        O = ret;
    end
    
    NN_train();
    printf('TRAINING COMPLETED! NOW PREDICTING.\n');
    prediction = NN_test();
end
