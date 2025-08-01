// assets/css/postcss.config.js
module.exports = {
    plugins: [
        // 使用 @tailwindcss/postcss 作为 PostCSS 插件
        require('@tailwindcss/postcss'),
        require('autoprefixer'),
    ]
}