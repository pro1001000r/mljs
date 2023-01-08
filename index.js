// https://qudata.com/ml/ru/NN_CNN_Explainable.html

console.log("первая нейронка"); //вывод

const SeedRandom = require("seedrandom")(34);

//******************************Работа с массивами *************************************
//функция создания массива
function matrixnew(row, column = 1, def = 0) {
  var A = new Array();
  //Указать количество строк
  A.length = row;
  //Указать количество столбцов
  for (var i = 0; i < row; i++) {
    // Создать подмассив в массиве
    A[i] = new Array();
    // Установить длину массива
    A[i].length = column;
  }

  //Заполнить массив A значениями 1
  for (var i = 0; i < A.length; i++)
    for (var j = 0; j < A[i].length; j++) {
      A[i][j] = def;
      if (def == 111) {
        A[i][j] = SeedRandom();
      }
    }
  //Вывести массив A в return
  return A;
}

//функция создания Вектора
function matrix1(el, def = 0) {
  var A = new Array();
  //Указать количество строк
  A.length = el;
  //Указать количество столбцов
  for (var i = 0; i < A.length; i++) {
    //Заполнить массив A значениями 1
    A[i] = def;
    if (def == 111) {
      A[i] = SeedRandom();
    }
  }

  //Вывести массив A в return
  return A;
}

//Функция умножения векторов
function multiply(W, X) {
  var sum = 0;
  for (var i = 0; i < X.length; i++) {
    sum += W[i] * X[i];
  }

  return sum;
}

//Входные даннные
let data = [
  // { input: [1, 0], output: 1 },
  // { input: [0, 0], output: 0 },
  // { input: [1, 0], output: 1 },
  // { input: [0, 1], output: 1 },
  // { input: [1, 1], output: 0 },
  { input: [4, 1], output: 1 },
  { input: [4, 2], output: 1 },
  { input: [7, 1], output: 1 },
  { input: [6, 7], output: 1 },
  { input: [8, 9], output: 0 },
  { input: [4, 12], output: 0 },
  { input: [0, 0], output: 1 },
  { input: [4, 0], output: 1 },
  { input: [3, 0], output: 1 },
  { input: [2, 10], output: 0 },
  { input: [4, 4], output: 1 },
  { input: [0, 5], output: 1 },
  { input: [-1, -5], output: 1 },
  { input: [-2, 15], output: 0 },
  { input: [-3, 5], output: 1 },
  { input: [-15, 55], output: 0 },
  { input: [-45, 85], output: 0 },
  { input: [-50, 5], output: 0 },
];

// Веса подобранные случайным образом
const weight2 = {
  i1_h1: 0,
  i1_h2: 0,

  i2_h1: 0,
  i2_h2: 0,

  h1_o1: 0,
  h2_o1: 0,

  bias_h1: 0,
  bias_h2: 0,
  bias_o1: 0,
};

const weight = {
  i1_h1: SeedRandom(),
  i1_h2: SeedRandom(),

  i2_h1: SeedRandom(),
  i2_h2: SeedRandom(),

  h1_o1: SeedRandom(),
  h2_o1: SeedRandom(),

  bias_h1: SeedRandom(),
  bias_h2: SeedRandom(),
  bias_o1: SeedRandom(),
};
//*********************************************************************************************************** */
//Функция активации (сигмоида)
const sigmoid3 = (x) => (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const sigmoid2 = (x) => {
  let y = x + 0.5;
  if (x > 0.5) {
    y = 1;
  }
  if (x < -0.5) {
    y = 0;
  }
  return y;
};
const sigmoid4 = (x) => {
  let y = x;

  if (x < 0) {
    y = 0;
  }
  return y;
};

//Фукция производной от сигмоиды для обучения обратное распространение ошибки весов
const p_sig3 = (x) => {
  const fx = sigmoid(x);
  return 1 - fx ** 2;
};
const p_sig = (x) => {
  const fx = sigmoid(x);
  return fx * (1 - fx);
};
const p_sig4 = (x) => {
  if (x > 0) {
    y = 1;
  } else {
    y = SeedRandom();
  }
  return y;
};

// let wh = matrixnew(2,2,111);
// console.log(w); //вывод
// let x = matrix1(2,111);
// console.log(x); //вывод
// let bh = matrix1(2,111);
// console.log(b); //вывод
// let sum = multiply(w[0],x) + b[0];
// console.log(sum); //вывод

const w = {
  wh: matrixnew(2, 2, 111),
  bh: matrix1(2, 111),
  wo: matrixnew(1, 2, 111),
  bo: matrix1(1, 111),
};

let wh = matrixnew(2, 2, 111);
let bh = matrix1(2, 111);
let wo = matrixnew(1, 2, 111);
let bo = matrix1(1, 111);

//выходной слой h
let h = matrix1(2);
let h_input = matrix1(2);
//выходной слой o
let o = matrix1(1);
let o_input = matrix1(1);

//новая нейронка на матрицах
function MNN(i, j) {
  var x = matrix1(2);
  x[0] = i;
  x[1] = j;

  h_input[0] = multiply(w.wh[0], x) + w.bh[0];
  h[0] = sigmoid(h_input[0]);

  h_input[1] = multiply(w.wh[1], x) + w.bh[1];
  h[1] = sigmoid(h_input[1]);

  o_input[0] = multiply(w.wo[0], h) + w.bo[1];
  o[0] = sigmoid(o_input[0]);

  return o[0];
}

//новая нейронка на матрицах
function Mtrain() {
  const w_d = {
    wh: matrixnew(2, 2),
    bh: matrix1(2),
    wo: matrixnew(1, 2),
    bo: matrix1(1),
  };

  //цикл по данным
  for (const {
    input: [i1, i2],
    output,
  } of data) {
    //Получаем заново нейронку для весов************************************
    var x = matrix1(2);
    x[0] = i1;
    x[1] = i2;

    h_input[0] = multiply(w.wh[0], x) + w.bh[0];
    h[0] = sigmoid(h_input[0]);

    h_input[1] = multiply(w.wh[1], x) + w.bh[1];
    h[1] = sigmoid(h_input[1]);

    o_input[0] = multiply(w.wo[0], h) + w.bo[0];
    o[0] = sigmoid(o_input[0]);
    //***************************************** */

    //ищем разницу между конечным результатом и вычесленным нейронкой
    const delta = output - o[0];

    //console.log(delta); //вывод
    //как бы возвращаемся назад
    const o1_d = delta * p_sig(o_input[0]);
    w_d.wo[0][0] += h[0] * o1_d;
    w_d.wo[0][1] += h[1] * o1_d;
    w_d.bo[0] += o1_d;

    const h0_d = o1_d * p_sig(h_input[0]);

    w_d.wh[0][0] += x[0] * h0_d;
    w_d.wh[0][1] += x[1] * h0_d;
    w_d.bh[0] += h0_d;

    const h1_d = o1_d * p_sig(h_input[1]);

    w_d.wh[1][0] += x[0] * h1_d;
    w_d.wh[1][1] += x[1] * h1_d;
    w_d.bh[1] += h1_d;
  }

  return w_d;
}

const w_d = Mtrain();
console.log(w_d); //вывод

//Обновляем веса в массиве
const MapplyTrainUpdate = (deltas = Mtrain()) => {
  // Object.keys(w).forEach((key) => {
  //   console.log(key); //вывод
  //   w[key] += deltas[key];
  // });
  for (var key in w) {


    //if w[key] == здесь закончил (нужно сопоставить матрицы) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  }
  console.log('обученный вывод'); //вывод
  console.log(w); //вывод
};

//Сама нейронка!!!!!!!!
const NN = (x1, x2) => {
  //вход на скрытый
  const h1_input = weight.i1_h1 * x1 + weight.i2_h1 * x2 + weight.bias_h1;
  const h1 = sigmoid(h1_input);

  const h2_input = weight.i1_h2 * x1 + weight.i2_h2 * x2 + weight.bias_h2;
  const h2 = sigmoid(h2_input);

  //скрытый на выход
  const o1_input = weight.h1_o1 * h1 + weight.h2_o1 * h2 + weight.bias_o1;
  const o1 = sigmoid(o1_input);

  return o1;
};

//Обучение
const train = () => {
  const w_d = {
    i1_h1: 0,
    i1_h2: 0,
    i2_h1: 0,
    i2_h2: 0,
    h1_o1: 0,
    h2_o1: 0,
    bias_h1: 0,
    bias_h2: 0,
    bias_o1: 0,
  };

  //цикл по данным
  for (const {
    input: [i1, i2],
    output,
  } of data) {
    //Получаем заново нейронку для весов************************************
    //вход на скрытый
    const h1_input = weight.i1_h1 * i1 + weight.i2_h1 * i2 + weight.bias_h1;
    const h1 = sigmoid(h1_input);

    const h2_input = weight.i1_h2 * i1 + weight.i2_h2 * i2 + weight.bias_h2;
    const h2 = sigmoid(h2_input);

    //скрытый на выход
    const o1_input = weight.h1_o1 * h1 + weight.h2_o1 * h2 + weight.bias_o1;
    const o1 = sigmoid(o1_input);
    //***************************************** */

    //получаем результат по нейронке
    //const o1 = NN(x1, x2);
    //ищем разницу между конечным результатом и вычесленным нейронкой
    const delta = output - o1;

    //console.log(delta); //вывод
    //как бы возвращаемся назад
    const o1_d = delta * p_sig(o1_input);
    w_d.h1_o1 += h1 * o1_d;
    w_d.h2_o1 += h2 * o1_d;
    w_d.bias_o1 += o1_d;

    const h1_d = o1_d * p_sig(h1_input);

    w_d.i1_h1 += i1 * h1_d;
    w_d.i2_h1 += i2 * h1_d;
    w_d.bias_h1 += h1_d;

    const h2_d = o1_d * p_sig(h2_input);

    w_d.i1_h2 += i1 * h2_d;
    w_d.i2_h2 += i2 * h2_d;
    w_d.bias_h2 += h2_d;
  }

  return w_d;
};

//Обновляем веса в массиве
const applyTrainUpdate = (deltas = train()) => {
  Object.keys(weight).forEach((key) => {
    weight[key] += deltas[key];
  });
};
//****************************************************************************************************************** */
//Вывод в консоль
const showResult = () => {
  data.forEach(({ input: [i1, i2], output: y }) => {
    let res = NN(i1, i2);

    let strres = "не подходит";
    if ((y == 0 && res < 0.2) || (y == 1 && res > 0.8)) {
      strres = "Ок!!!";
    }

    console.log(
      i1 + " и " + i2 + " результат: " + res + " => " + y + "    " + strres
    ); //вывод
  });
};

//Вывод в консоль ТОЧЕЧНАЯ!!!!!!!!!!!!!!!!!!!!!!1
const show = (i1, i2) => {
  let res = NN(i1, i2);

  let strres = "не подходит";
  if (res < 0.2 || res > 0.8) {
    strres = "Ок!!!";
    if (res < 0.2) {
      strres += " ЛОЖЬ";
    } else {
      strres += " ИСТИНА";
    }
  }
  console.log("Вывод: " + i1 + " и " + i2 + " => " + res + " " + strres); //вывод
};

console.log("--------------------Первоначальная---------------------"); //вывод
applyTrainUpdate();

showResult();
console.log("------------------------Окончательный-----------------"); //вывод

// Само обучение сети!!!!!!!!!!!!!!!!!!!!!!
for (let i = 0; i < 10000; i++) {
  applyTrainUpdate();
}
showResult();
//console.log(weight); //вывод

show(4, 4);
show(2, 3);
show(4, 1);
show(3, 5);
show(1, 30);
show(-6, 30);
show(7, 7);
