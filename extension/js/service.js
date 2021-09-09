export default {
    getArticleOpinion({ body, algorithmName }) {
        return fetch(`http://127.0.0.1:5000/${algorithmName}`, {
            method: 'POST',
            headers: { "Accept": "application/json" },
            body: JSON.stringify(body)
        })
            .then(res => res.json())
            .then(res => res)
            .catch(err => console.log(err))
    }
}