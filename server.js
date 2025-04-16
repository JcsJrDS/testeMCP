const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware para processar dados do formulário
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Servir arquivos estáticos
app.use(express.static(path.join(__dirname)));

// Rota principal
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Rota para processar o formulário de contato
app.post('/api/contato', (req, res) => {
    const { nome, email, mensagem } = req.body;
    
    // Aqui você pode adicionar código para salvar em um banco de dados
    // ou enviar e-mail com os dados do formulário
    
    console.log('Dados do formulário recebidos:');
    console.log(`Nome: ${nome}`);
    console.log(`Email: ${email}`);
    console.log(`Mensagem: ${mensagem}`);
    
    // Responder ao cliente
    res.status(200).json({ 
        success: true, 
        message: 'Mensagem recebida com sucesso! Entraremos em contato em breve.'
    });
});

// Iniciar o servidor
app.listen(PORT, () => {
    console.log(`Servidor rodando na porta ${PORT}`);
    console.log(`Acesse: http://localhost:${PORT}`);
}); 