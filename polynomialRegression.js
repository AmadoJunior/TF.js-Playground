let x_vals = [];
let y_vals = [];

let a, b, c;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup(){
    createCanvas(400, 400);
    background(0);

    a = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));
}

function mousePressed(){
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
}

function predict(x){
    const xs = tf.tensor1d(x);
    //y = ax^2 + bx + c
    const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
    return ys;
}

function loss(predictions, labels){
    return predictions.sub(labels).square().mean();
}

function draw(){
    tf.tidy(() => {
        if(x_vals.length > 0){
            //Minimize Loss Fn
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
        }

    })
    
    //Render
    background(0);
    stroke(255);
    strokeWeight(5);

    for(let i = 0; i < x_vals.length; i++){
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }

    let curveX = [];
    for(let x = -1; x < 1; x += 0.05){
        curveX.push(x);
    }

    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for(let i = 0; i < curveX.length; i++){
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], -1, 1, height, 0);
        vertex(x,y);
    }
    endShape();

    console.log(tf.memory().numTensors);
}