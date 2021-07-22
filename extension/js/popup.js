document.addEventListener('DOMContentLoaded', init);

function detectFakeNews() {
  console.info("DETECTING!!! PLEASE WAIT!");
  chrome.runtime.sendMessage({ url: window.location.href }, function(response) {
    console.log(response.farewell);
  });
}


function showMessage(message) {
  const safe = document.getElementById("safe");
  const danger = document.getElementById("danger");

  console.log(message)
  // reset elements
  if (!safe.classList.contains('hidden')) {
    safe.classList.add('hidden')
  }

  if (!danger.classList.contains('hidden')) {
    danger.classList.add('hidden')
  }

  if (message) {
    safe.classList.remove('hidden')
  } 

  if (!message) {
    danger.classList.remove('hidden')
  }
}

function init(event) {
  const action = document.getElementById("action");

  action.addEventListener("click", async () => {
    // get current tab
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // inject script into current page
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: detectFakeNews,
    });
  });

  // listen for message from `detectFakeNews` injected script
  chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
      fetch('http://127.0.0.1:5000/keras', {
        method: 'POST', 
        headers: { "Accept": "application/json" },
        body: JSON.stringify(request.url)
      })
      .then((res) => res.json())
      .then((res) => showMessage(res.legit))
      .catch((err) => console.log(err))

      if (request.greeting == "hello")
        sendResponse({farewell: "goodbye"});
    }
  );
}