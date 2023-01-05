console.log("первая нейронка"); //вывод

const SeedRandom = require("seedrandom")(34);

//Входные даннные
let data = [
  { input: [5, 2], output: 1 },
  { input: [5, 4], output: 1 },
  { input: [1, 4], output: 1 },
  { input: [1, 2], output: 1 },
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
const weight = {
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

const weight2 = {
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
  
//Функция активации (сигмоида)
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
//Фукция производной от сигмоиды для обучения обратное распространение ошибки весов
const p_sig = (x) => {
  const fx = sigmoid(x);
  return fx * (1 - fx);
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
          strres += ' ЛОЖЬ';
      }else{
          strres += ' ИСТИНА';
      }
    }
  
    console.log(
      "Вывод: " + i1 + " и " + i2 + " --- " + res + " должно быть " + strres
    ); //вывод
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

console.log("--------------------Первоначальная---------------------"); //вывод
applyTrainUpdate();
showResult();
console.log("------------------------Окончательный-----------------"); //вывод

//Само обучение сети!!!!!!!!!!!!!!!!!!!!!!
for (let i = 0; i < 100000; i++) {
  applyTrainUpdate();
}
showResult();
//console.log(weight); //вывод

show(4,4);
show(2,3);
show(4,1);
show(3,5);
show(1,30);
show(-6,30);
show(7,7);