clc;
clear ;
close all;
%% Train
cd('Data')
data_Train = readtable('Train_Arabic_Digit.txt','TextType','string');
data=( table2array(data_Train))';
[m,n]=size(data);
k=1;
j=1;
for i=1:n 
    if isnan(data(:,i))
        j=j+1;
        k=1;
    else
        x{j,k}=data(:,i);
        k=k+1;
    end  
end

[n,m]=size(x);
for i = 1:n
       for j= 1:m
           a=x{i,j};
           if isempty(a)
               break
           end
           if j==1
           x_train{i,1}=a;
           else
               b=horzcat(x_train{i,1},a);
               x_train{i,1}=b;
           end
       end
end

label=cell(1,660);
i=0;
for k=1:660:6600
    k=k-1;
        for j=1:660
         label(1,j)={i};
         y_train((k+j),1)=label(1,j);
        end
        i=i+1;
end

numObservations = numel(x_train);
for i=1:numObservations
    sequence = x_train{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,I] = sort(sequenceLengths);
x_train = x_train(I);
y_train = y_train(I);
y_train = categorical(string(y_train));

figure;
bar(sequenceLengths);
ylim([0 100]);
xlabel("Sequence");
ylabel("Length");
title("Sorted Train Data");

%% Test

data_test = readtable('Test_Arabic_Digit.txt','TextType','string');
data=( table2array(data_test))';
[m,n]=size(data);
k=1;
j=1;
x=[ ];
for i=1:n 
    if isnan(data(:,i))
        j=j+1;
        k=1;
    else
        x{j,k}=data(:,i);
        k=k+1;
    end  
end

[n,m]=size(x);
for i = 1:n
       for j= 1:m
           a=x{i,j};
           if isempty(a)
               break
           end
           if j==1
           x_test{i,1}=a;
           else
               b=horzcat(x_test{i,1},a);
               x_test{i,1}=b;
           end
       end
end

label=cell(1,220);
i=0;
for k=1:220:2200
    k=k-1;
        for j=1:220
         label(1,j)={i};
         y_test((k+j),1)=label(1,j);
        end
        i=i+1;
end

sequence=[];
sequenceLengths=[];
numObservations = numel(x_test);
for i=1:numObservations
    sequence = x_test{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,I] = sort(sequenceLengths);
x_test = x_test(I);
y_test = y_test(I);
y_test = categorical(string(y_test));

figure;
bar(sequenceLengths);
ylim([0 100]);
xlabel("Sequence");
ylabel("Length");
title("Sorted Test Data");
%% Training
inputSize=13;
numHiddenUnits=60;
numClasses=10;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 10;
miniBatchSize = 132;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(x_train,y_train,layers,options);

y_out = classify(net,x_test, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

accuracy = (sum(y_out == y_test)./numel(y_test))*100



