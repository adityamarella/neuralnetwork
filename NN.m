function prediction = NN(train, dev, learning_factor, max_iterations, nin, nhidden, nout, Wih, Who)


    function [Oh Oo] = update_output(X, Oh, Oo)
        %calculating the output for the hidden nodes
        %input -> O(1,nin)
        for i = 1:nhidden
            Oh(i) = sigmoid(dot(transpose(Wih(1:nin+1,i)), X));
        end

        %calculating the output values for the output nodes
        %input -> O(nin+1, nhidden)
        for i = 1:nout
            Oo(i) = sigmoid(Who(1,i) + dot(transpose(Who(2:nhidden+1,i)), Oh));
        end
    end

    function [Eh Eo] = backpropagate_error(y, Oh, Oo)
        %propagate error for output units
        for i=1:nout
            o = Oo(i);
            Eo(i) = o * (1-o) * (y-o);
        end

        %propagate error for hidden units
        for i=1:nhidden+1
            if i==1
                o = 1;
            else
                o = Oh(i-1);
            end
            Eh(i) = o * (1-o) * dot(Who(i,:), Eo);
        end
    end

    function ret = update_weights(X, Oh, Oo, Eh, Eo)
        tmp = [1 Oh(1:nhidden)];
        for i=1:nin+1
            for j=1:nhidden
                Wih(i,j) = Wih(i,j) + learning_factor*Eh(j+1)*X(i);
            end
        end
        for i=1:nhidden+1
            for j=1:nout
                Who(i,j) = Who(i,j) + learning_factor*Eo(j)*tmp(i);
            end
        end
    end

    function ret = NN_train()

        [rows cols] = size(train);
        Oh = zeros(1,nhidden);
        Oo = zeros(1,nout);

        Eh = zeros(1,nhidden+1);
        Eo = zeros(1,nout);

        %Backpropagation with one hidden layer 
        %Stochastic gradient descent to update weights
        for iteration = 1:max_iterations
            sigma = 0;
            for row = 1:rows
                X = [train(row,1:cols-1)];
                y = train(row,cols);
               
                %X0 is set to 1 
                X = [1 X(1:cols-1)];

                [Oh Oo] = update_output(X, Oh, Oo);
                [Eh Eo] = backpropagate_error(y, Oh, Oo, Eh, Eo);
                update_weights(X, Oh, Oo, Eh, Eo);

                for i=1:nout
                    sigma = sigma + 0.5*(y-Oo(i))*(y-Oo(i));
                end
            end
            if mod(iteration, 10) == 0
                printf('%f\n', sigma);
            end
        end
    end
    
    function O = NN_test()
        
        [rows cols] = size(dev);
        Oh = zeros(1,nhidden);
        Oo = zeros(1,nout);

        ret = zeros(1,rows);
        for i = 1:rows
            X = dev(i,:);
            X = [1 X(1:cols)];
            [Oh, Oo] = update_output(X, Oh, Oo);
            ret(i) = Oo(1);
        end
        O = ret;
    end
    
    NN_train();
    printf('TRAINING COMPLETED! NOW PREDICTING.\n');
    prediction = NN_test();
end
