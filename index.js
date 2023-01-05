console.log("первая нейронка"); //вывод

const SeedRandom = require('seedrandom')(34);
//Входные даннные

let data = [
  { input: [0, 0], output: 0 },
  { input: [1, 0], output: 1 },
  { input: [0, 1], output: 1 },
  { input: [1, 1], output: 0 },
];

const weight = {
    i1_h1:SeedRandom(),
    i1_h2:SeedRandom(),
    i2_h1:SeedRandom(),
    i2_h2:SeedRandom(),
    h1_o1:SeedRandom(),
    h2_o1:SeedRandom(),
};
console.log(weight); //вывод