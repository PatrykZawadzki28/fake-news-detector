chrome.runtime.onInstalled.addListener(function() {
    chrome.contextMenus.create({
        id: "auth-checker",
        title: "Check authenticity",
        contexts:["selection"],
    });
});

chrome.contextMenus.onClicked.addListener(function(info, tab) {
    const { menuItemId, selectionText } = info;

    if (menuItemId == "auth-checker") {
        chrome.tabs.create({url:"popup.html"});
        chrome.runtime.sendMessage({ type: 'fetch', text: selectionText });
    }
});

// chrome.contextMenus.create({
//     title: "Test %s menu item",
//     contexts:["selection"],
//     onclick: function(info, tab) {
//         sendSearch(info.selectionText);
//     }
// });

// chrome.extension.onRequest.addListener(function (request) {
//     var tx = request;
//     var title = "Test '" + tx + "' menu item";
//     var id = chrome.contextMenus.create({"title": title, "contexts":["selection"],
//         "onclick": sendSearch(tx)});
//     console.log("selection item:" + id);
// });
//
// getSelection(function(tx) {
//     var title = "Test '" + tx + "' menu item";
//     var id = chrome.contextMenus.create({"title": title, "contexts":["selection"],
//         "onclick": function(info, tab) {
//             sendSearch(info.selectionText);
//         }});
//     console.log("selection item:" + id);
// })