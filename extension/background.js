import Service from './js/service.js';

function detectFakeNews() {
    console.info("DETECTING!!! PLEASE WAIT!");
    chrome.runtime.sendMessage({ type: 'fetch', url: window.location.href }, function(response) {
        console.log(response.farewell);
    });
}

const resetMessages = ({ preload, response }) => {
    if (!preload.classList.contains('hidden')) {
        preload.classList.add('hidden')
    }
    response.innerHTML = ''
}

const showMessage = ({ preload, safe, danger, response }, answer) => {
    Object.entries(answer).forEach(([key, value]) => {
        if (value != null) {
            const elementBox = document.createElement('div')
            const name = document.createElement('h5')
            const iconBox = document.createElement('div')
            const icon = document.createElement('i')

            elementBox.className = 'row'
            name.className = 'col s6'
            name.innerText = key
            iconBox.className = 'col s3'
            icon.className = 'medium material-icons green-text'
            icon.innerText =  value > 0 ? 'thumb_up' : 'thumb_down'

            elementBox.appendChild(name)
            elementBox.appendChild(iconBox)
            iconBox.appendChild(icon)

            response.appendChild(elementBox)
        }
    })

    if (!answer) {
        danger.classList.remove('hidden');
    }
}

function init(event) {
    const action = document.getElementById("action");
    const preload = document.getElementById("preloader");
    const safe = document.getElementById("safe");
    const danger = document.getElementById("danger");
    const response = document.getElementById("response");

    // get algorithm switch nodeElements;
    const lstm = document.getElementById("lstm");
    const nlp = document.getElementById("nlp");
    const lstm_simple = document.getElementById("lstm_simple");

    let algorithms = {
        lstm: true,
        nlp: false,
        lstm_simple: false
    }

    // set algorithms to use
    Array.from([lstm, nlp, lstm_simple]).forEach((el) => {
        el.addEventListener('change', () => {
            algorithms[el.id] = el.checked;
        })
    });

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
            console.log('is fetching!!1')

            const { type, text, url } = request;

            if (type === 'fetch') {
                resetMessages({preload, response});
                console.log('is fetching!!')
                preload.classList.remove('hidden');

                Service.getArticleOpinion({body: { text: text || '', url: url || '', algorithms}, algorithmName: 'detect'})
                    .then(res => {
                        const { answer } = res;
                        preload.classList.add('hidden');
                        showMessage({preload, safe, danger, response}, answer)
                    })
            }
        }
    );
}

document.addEventListener('DOMContentLoaded', init);
