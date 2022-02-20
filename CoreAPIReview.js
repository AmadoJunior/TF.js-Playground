function setup(){
    noCanvas();
    // Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
    const a = tf.tensor([[1, 2], [3, 4]]);
    console.log('shape:', a.shape);
    a.print();

    // Or you can create a tensor from a flat array and specify a shape.
    const shape = [2, 2];
    const b = tf.tensor([1, 2, 3, 4], shape);
    console.log('shape:', b.shape);
    b.print();

    //By default, tf.Tensors will have a float32 dtype. tf.Tensors can also be created with bool, int32, complex64, and string dtypes:
    const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
    console.log('shape:', a.shape);
    console.log('dtype', a.dtype);
    a.print();

    //Generate array of 15 random values
    const values = [];
    for (let i = 0; i < 15; i++){
        values[i] = random(0,100);
    }

    //5 row, 3 column
    const shape = [5, 3];

    //create tensor
    const data = tf.tensor(values, shape);

    //log
    console.log(data.toString());
    console.log(data);
}

function draw(){
    //tidy: Executes the provided function fn and after it is executed, cleans up all intermediate tensors allocated by fn except those returned by fn. fn must not return a Promise (async functions not allowed). The returned result can be a complex object.
    tf.tidy(() => {
        //Generate array of 15 random values
        const values = [];
        for (let i = 0; i < 15; i++){
            values[i] = random(0,100);
        }

        //shapeA: 5 row, 3 column
        const shape = [5, 3];

        //create tensors
        const a = tf.tensor(values, shape, "int32");
        const b = tf.tensor(values, shape, "int32");

        //transpose b to allow for matrix mult
        const bTransposed = b.transpose();

        //log
       const v = tf.matMul(a, bTransposed);
       v.print();
    })

    //log num of tensors
    console.log(tf.memory().numTensors);
}