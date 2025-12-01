# Intrepretability

Results : 


LIME : 

| Feature            | Contribution |
| ------------------ | ------------ |
| **capital.gain**   | **+0.2731**  |
| **education.num**  | +0.0996      |
| **hours.per.week** | +0.0731      |
| **age**            | +0.0727      |
| **capital.loss**   | +0.0537      |
| **fnlwgt**         | +0.0057      |




CAPTUM : 

Integrated Gradient 

| Feature            | Attribution |
| ------------------ | ----------- |
| **age**            | **−0.2311** |
| **fnlwgt**         | +0.0481     |
| **education.num**  | +0.0185     |
| **capital.gain**   | +0.1379     |
| **capital.loss**   | +0.0059     |
| **hours.per.week** | +0.0217     |


SmoothGrad

| Feature            | Attribution |
| ------------------ | ----------- |
| **age**            | **−0.2169** |
| **fnlwgt**         | +0.0368     |
| **education.num**  | +0.0230     |
| **capital.gain**   | +0.1265     |
| **capital.loss**   | +0.0134     |
| **hours.per.week** | +0.0208     |
