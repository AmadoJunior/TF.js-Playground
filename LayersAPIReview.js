//Define Sequential Model
const model = tf.sequential();

//Define Configuration for Layers
const configHidden = {
    units: 4, //Units
    inputShape: [2], //Input Shape
    activation: "sigmoid" //Activation Fn
}

const configOutput = {
    units: 1, //Units
    // inputShape: [4], Can Inferred from Prev Layer
    activations: "sigmoid" //Activation Fn
}

//Create Layer (Dense)
const hidden = tf.layers.dense(configHidden);
const output = tf.layers.dense(configOutput);

//Add Layers to Model
model.add(hidden);
model.add(output);

//Define Optimizer && Loss Functions
const sgdLearningRate = 0.5;
const sgdOpt = tf.train.sgd(sgdLearningRate);
const lossFn = "meanSquaredError";

//Define Model Configuration Object
const config = {
    optimizer: sgdOpt,
    loss: lossFn
}

//Compile Model
model.compile(config);

//Example Training Data
const xs = tf.tensor2d([
    [0, 0],
    [1, 1],
    [0.5, 0.5],
]);

const ys = tf.tensor2d([
    [1],
    [0.5],
    [0],
]);

//Optional
const options = {
    verbose: true,
    shuffle: true, //Shuffles training data 
    batchSize: 32, //batchSize (number) Number of samples per gradient update. If unspecified, it will default to 32.
    epochs: 10 //epochs (number) Integer number of times to iterate over the training data arrays.
};

async function train() {
    for(let i = 0; i < 1000; i++){
        const res = await model.fit(xs, ys, options);
        console.log(res.history.loss[0]);
    }
}

train().then(() => {
    console.log("Training Complete.");
    const outputs = model.predict(xs);
    outputs.print();
});



