const inputbox = document.querySelector("#input-box");
const ptag = document.querySelector("#result");

const handleClick = () => {
  text = inputbox.value;
  var response = fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },

    body: JSON.stringify({ news: text }),
  })
    .then((response) => response.json())
    .then((result) => {
      return result.pred;
    });

  response.then((data) => {
    if (data == 1) {
      ptag.innerHTML = "This True news";
    } else ptag.innerHTML = "This Fake news";
  });
};
