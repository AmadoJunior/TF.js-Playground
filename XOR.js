//TF Model
let model;
//P5 Vars
let resolution = 40;
let cols;
let rows;

//Temp
let xs;

const train_xs = tf.tensor2d([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])

const train_ys = tf.tensor2d([
    [0],
    [1],
    [1],
    [0],
])


function setup(){
    //Setup Canvas
    createCanvas(400, 400);
    cols = width / resolution;
    rows = height / resolution;

    //Create the Input Data
    let inputs = [];
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;
            inputs.push([x1, x2]);
        }
    }

    xs = tf.tensor2d(inputs);

    //Define myModel
    myModel = tf.sequential();

    //Create Layers (Dense)
    const hidden = tf.layers.dense({
        inputShape: [2],
        units: 2,
        activation: "sigmoid"
    });

    const output = tf.layers.dense({
        units: 1,
        activation: "sigmoid"
    });

    //Add Layers
    myModel.add(hidden);
    myModel.add(output);

    //Define Optimizer && Loss Functions
    const sgdLearningRate = 0.5;
    const sgdOpt = tf.train.sgd(sgdLearningRate);
    const lossFn = "meanSquaredError";

    //Compile myModel using Configuration Object derived from prev variables
    myModel.compile({
        optimizer: sgdOpt,
        loss: lossFn
    });

    setTimeout(train, 10);
}

function trainModel() {
    //Returns Promise from fit() function
    return myModel.fit(train_xs, train_ys, {
        suffle: true,
        epochs: 10
    })
}

function train(){
    trainModel().then((result) => {
        console.log(result.history.loss[0]);
        setTimeout(train, 10);
    })
}

function draw(){
    background(0);    

    tf.tidy(() => {
        //Get Predictions
        let ys = myModel.predict(xs);
        let y_values = ys.dataSync();

        //Draw Results
        let index = 0;
        for (let i = 0; i < cols; i++) {
          for (let j = 0; j < rows; j++) {
            let br = y_values[index] * 255;
            fill(br);
            rect(i * resolution, j * resolution, resolution, resolution);
            fill(255 - br);
            textSize(8);
            textAlign(CENTER, CENTER);
            text(
              nf(y_values[index], 1, 2),
              i * resolution + resolution / 2,
              j * resolution + resolution / 2
            );
            index++;
          }
        }
    })
}